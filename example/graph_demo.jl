using GLVisualize, GeometryTypes, Colors, Reactive
using GLAbstraction
using LightGraphs, GraphLayout, GeometryTypes

w = glscreen();@async renderloop(w)
function generate_spikes(v0::Vector{Float32}, t::Float32)
    for i=1:length(v0)
        v = v0[i]
        damper = -sign(v)*v*0.01
        s = -sign(v)*(sin(t-(i/length(v0))+(rand()/4)) + 1) / 1.5
        x = rand() * s
        v0[i] = v + x
    end
    v0
end
function mix_linearly{C<:Colorant}(a::C, b::C, s)
    RGBA{Float32}((1-s)*comp1(a)+s*comp1(b), (1-s)*comp2(a)+s*comp2(b), (1-s)*comp3(a)+s*comp3(b), 1.)
end
function color_lookup(value, cmap, color_norm)
    mi,ma = color_norm
    scaled = clamp((value-mi)/(ma-mi), 0, 1)
    index = scaled * (length(cmap)-1)
    i_a, i_b = floor(Int, index)+1, ceil(Int, index)+1
    mix_linearly(cmap[i_a], cmap[i_b], scaled)
end
function gen_color(vec, cmap, color_norm)
    RGBA{Float32}[color_lookup(x, cmap, color_norm) for x in vec]
end
function gen_hist(v0, val)
    for i=1:length(v0)
        v0[i] = circshift(v0[i], 1)
        v0[i][1] = val[i]
    end
    v0
end
g = RoachGraph(30)
adj_matrix = adjacency_matrix(g)
N = size(adj_matrix, 1)

const t = bounce(linspace(0f0,500f0, 1000))
cmap = map(RGBA{Float32}, map(RGBA{Float32}, colormap("RdBu", 5)))
colornorm = Signal(Vec2f0(-4f0, 4f0))
push!(colornorm, Vec2f0(-2,2))
signals = foldp(generate_spikes, rand(Float32, N)*0.1f0, t)
colormap_single = const_lift(gen_color, signals, cmap, colornorm)
hist_length = 200
history = foldp(gen_hist, [zeros(Float32, hist_length) for i=1:N], signals)
colors = foldp(gen_hist, [zeros(RGBA{Float32}, hist_length) for i=1:N], colormap_single)
gap = Signal(1.0f0)
range = linspace(-4f0, 4f0, hist_length)
selected_nodes = Signal(fill(false, N))
function outlinecolor(selection)
    println("outlinecolor")
    [s ? RGBA{Float32}(0.9,0.9,1,1.0) : RGBA{Float32}(0,0,0,0) for s in selection]
end
line_c = map(1:N) do i
    p = map(history) do ps
        Point2f0[xy for xy in zip(range, ps[i])]
    end
    c = const_lift(getindex, colors, i)
    visualize(p, :lines, color=c).children[]
end
function display_signals(nodes, g)
    next_position = 0.0
    for (line, selected) in zip(line_c, nodes)
        if selected
            line[:visible] = true
            line[:model] = translationmatrix(Vec3f0(0,0,next_position))
            next_position += g
        else
            line[:visible] = false
        end
    end
end
preserve(map(display_signals, selected_nodes, gap))

function visualize_graph(g, intensity, w, cmap, cnorm)
    adj_matrix = full(adjacency_matrix(g))
    N = size(adj_matrix, 1);
    locs = 2f0*rand(Point3f0, N) .- 1f0
    GraphLayout.layout_spring!(adj_matrix, locs)
    edg = edges(g)
    x = collect(edg)
    indices = Array(Int, length(x)*2)

    for (i, e) in enumerate(x)
        indices[((i-1)*2)+1] = first(e)
        indices[((i-1)*2)+2] = last(e)
    end
    indices_s = Signal(indices)
    outline_color = map(outlinecolor, selected_nodes)

    nodes = visualize(
        (Circle(Point2f0(0), 0.1f0), locs), billboard=true,
        color=colormap_single, indices=map(unique, indices_s),
        stroke_color=outline_color, stroke_width=0.001f0
    ).children[]
    view(nodes, w, camera=:perspective)

    gpu_points = w.renderlist[1][1][:position]

    verts = visualize(gpu_points, :linesegment, indices=indices_s, color=colormap_single)
    view(verts, w, camera=:perspective)
    nodes, verts.children[], indices_s
end
using GLWindow
area1, area51 = x_partition(w.area, 50.)
@materialize mouse_buttons_pressed, mouseposition, buttons_pressed = w.inputs
cam_action = const_lift(==, Set([GLFW.KEY_LEFT_CONTROL]), buttons_pressed)
screena = Screen(w, area=area1)
screenb = Screen(w, area=area51)
camera = PerspectiveCamera(
    screena.inputs, Vec3f0(3), Vec3f0(0),
    keep=map(AND, cam_action, screena.inputs[:mouseinside])
)
screena.cameras[:perspective] = camera

cubecamera(screenb)


point_robj, line_robj, indices_s = visualize_graph(g, signals, screena, cmap, colornorm)

view(line_c, screenb, camera=:perspective)


const gpu_position = point_robj[:position]
const m2id = mouse2id(w)
# interaction


isoverpoint = const_lift(is_same_id, m2id, point_robj)
# single left mousekey pressed (while no other mouse key is pressed)
left_pressed = const_lift(GLAbstraction.singlepressed, mouse_buttons_pressed, GLFW.MOUSE_BUTTON_LEFT)
# righ
right_pressed = const_lift(GLAbstraction.singlepressed, mouse_buttons_pressed, GLFW.MOUSE_BUTTON_RIGHT)
# dragg while key_pressed. Drag only starts if isoverpoint is true
mousedragg  = GLAbstraction.dragged(mouseposition, left_pressed, isoverpoint)

# use mousedrag and mouseid + index to actually change the gpu array with the positions
function apply_drag(v0, dragg)
    if !value(cam_action)
        if dragg == Vec2f0(0) # if drag just started. Not the best way, maybe dragged should return a tuple of (draggvalue, started)
            id, index = value(m2id)
            if id==point_robj.id && length(gpu_position) >= index
                prj_view = value(camera.projectionview)
                p0 = Point4f0(prj_view * Vec4f0(gpu_position[index], 1))
            else
                p0 = v0[3]
            end
        else
            id, index, p0 = v0
            if id==point_robj.id && length(gpu_position) >= index
                prj_view_inv = inv(value(camera.projectionview))
                area = value(camera.window_size)
                cam_res = Vec2f0(widths(area))
                dragg_clip_space = (Vec2f0(dragg)./cam_res) * p0[4] * 2
                pos_clip_space = p0 + Point4f0(dragg_clip_space, 0, 0)
                p_world_space = Point3f0(prj_view_inv * Vec4f0(pos_clip_space))
                gpu_position[index] = p_world_space
            end
        end
        return id, index, p0
    end
    v0
end

function remove_node(rp)
    if !value(cam_action)
        id, index = value(m2id)
        if rp && id==point_robj.id && length(gpu_position) >= index
            new_indices = Int[]
            indice_val = value(indices_s)
            for i=1:2:length(indice_val) #filter out indices
                a, b = indice_val[i], indice_val[i+1]
                if a != index && b != index
                    push!(new_indices, a, b)
                end
            end
            push!(indices_s, new_indices) # update indices!
        end
    end
    nothing
end
function select_nodes(mouse_pressed)
    mouse_pos = value(mouseposition)
    if mouse_pressed && isinside(value(screena.area), value(mouseposition)...)
        id, index = value(m2id)
        mods = value(buttons_pressed)
        selection = value(selected_nodes)
        should_append = Set([GLFW.KEY_LEFT_CONTROL]) == mods
        if !should_append # if we don't append, we at least need to clear the selection
            fill!(selection, false)
        end
        if id==point_robj.id && length(gpu_position) >= index
            if  should_append || isempty(mods) # now strg adds
                selection[index] = !selection[index]
            end
        end
        push!(selected_nodes, selection)
    end
    nothing
end
preserve(map(select_nodes, left_pressed))

preserve(foldp(apply_drag, (value(m2id)..., Point4f0(0)), mousedragg))
# On right click remove nodes!
preserve(map(remove_node, right_pressed))
