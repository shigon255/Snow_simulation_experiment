import taichi as ti
import numpy as np

# modified from mpm3d_ggui.py

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch, debug = True)

# basic parameters
# dim, n_grid, steps, dt = 2, 128, 20, 2e-4
#dim, n_grid, steps, dt = 2, 256, 32, 1e-4
dim, n_grid, steps, dt = 3, 64, 25, 1e-4
# dim, n_grid, steps, dt = 3, 64, 25, 2e-4
#dim, n_grid, steps, dt = 3, 128, 5, 1e-4

n_particles = n_grid**dim // 2**(dim - 1)
dx = 1 / n_grid

p_rho = 4 * 1e2
p_vol = (dx * 0.5)**2 # volume
p_mass = p_vol * p_rho
GRAVITY = [0, -9.8, 0]
bound = 3

# Material parameters(Young's modulus, poisson's ratio, ...)

# original mpm3d_ggui.py's parameter
# E = 1000  # Young's modulus
# nu = 0.2  #  Poisson's ratio

# snow2013 parameters
LOWER_HARDENING = 0
LOWER_YOUNG_MODULUS = 1
LOWER_CRITICAL_COMPRESSION = 2
REFERENCE = 3
LOWER_CRITICAL_COMPRESSION_STRETCH = 4
LOWER_CRITICAL_STRETCH = 5

TEST_TYPE = REFERENCE

NORMAL_E = 1.4 * 1e5
LOWER_E = 4.8 * 1e4
NORMAL_critical_compression = 2.5 * 1e-2
LOWER_critical_compression = 1.9 * 1e-2
NORMAL_critical_stretch = 7.5 * 1e-3
LOWER_critical_stretch = 5 * 1e-3
NORMAL_hardening = 10
LOWER_hardening = 5

ply_directory = ""

if TEST_TYPE == LOWER_HARDENING:
    E = NORMAL_E
    critical_compression = NORMAL_critical_compression
    critical_stretch = NORMAL_critical_stretch
    hardening = LOWER_hardening
    ply_directory = "lower_hardening_ply"
elif TEST_TYPE == LOWER_YOUNG_MODULUS:
    E = LOWER_E
    critical_compression = NORMAL_critical_compression
    critical_stretch = NORMAL_critical_stretch
    hardening = NORMAL_hardening
    ply_directory = "lower_young_modulus_ply"
elif TEST_TYPE == LOWER_CRITICAL_COMPRESSION:
    E = NORMAL_E
    critical_compression = LOWER_critical_compression
    critical_stretch = NORMAL_critical_stretch
    hardening = NORMAL_hardening
    ply_directory = "lower_critical_compression_ply"
elif TEST_TYPE == REFERENCE:
    E = NORMAL_E
    critical_compression = NORMAL_critical_compression
    critical_stretch = NORMAL_critical_stretch
    hardening = NORMAL_hardening
    ply_directory = "reference_ply"
elif TEST_TYPE == LOWER_CRITICAL_COMPRESSION_STRETCH:
    E = NORMAL_E
    critical_compression = LOWER_critical_compression
    critical_stretch = LOWER_critical_stretch
    hardening = NORMAL_hardening
    ply_directory = "lower_critical_compression_and_stretch_ply"
elif TEST_TYPE == LOWER_CRITICAL_STRETCH:
    E = NORMAL_E
    critical_compression = NORMAL_critical_compression
    critical_stretch = LOWER_critical_stretch
    hardening = NORMAL_hardening
    ply_directory = "lower_critical_stretch_ply"

nu = 0.2

mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))  # Lame parameters

# MPM and MLS-MPM

MPM = 0
MLS_MPM = 1

MPM_TYPE = MLS_MPM
# MPM_TYPE = MPM

mpm_directory = ""
if MPM_TYPE == MPM:
    mpm_directory = "MPM_ply"
elif MPM_TYPE == MLS_MPM:
    mpm_directory = "MLS_MPM_ply"


F_x = ti.Vector.field(dim, float, n_particles) # position
F_v = ti.Vector.field(dim, float, n_particles) # velocity
F_C = ti.Matrix.field(dim, dim, float, n_particles) # Affine matrix
F_dg = ti.Matrix.field(3, 3, dtype=float, 
                       shape=n_particles)  # deformation gradient
F_Jp = ti.field(float, n_particles) # det(Fp)

F_colors = ti.Vector.field(4, float, n_particles) 
F_colors_random = ti.Vector.field(4, float, n_particles) 
F_materials = ti.field(int, n_particles)
F_grid_v = ti.Vector.field(dim, float, (n_grid, ) * dim)
F_grid_m = ti.field(float, (n_grid, ) * dim)
F_grid_f = ti.Vector.field(dim, float, (n_grid, ) * dim)
F_used = ti.field(int, n_particles) # ?

neighbour = (3, ) * dim

WATER = 0
JELLY = 1
SNOW = 2

vec3 = ti.types.vector(3, float) # taichi type definition

# physical environment setting
center = (0.45, 0.3, 0.2)
r = 0.1
vertices = ti.Vector.field(dim, float, 8)
vertices_list = [[center[0] - r, center[1] - r, center[2] - r],
                  [center[0] - r, center[1] - r, center[2] + r],
                  [center[0] - r, center[1] + r, center[2] - r],
                  [center[0] - r, center[1] + r, center[2] + r],
                  [center[0] + r, center[1] - r, center[2] - r],
                  [center[0] + r, center[1] - r, center[2] + r],
                  [center[0] + r, center[1] + r, center[2] - r],
                  [center[0] + r, center[1] + r, center[2] + r]]

rotateaxis = ti.Vector([1.0, 1.0, 1.0])
angle = ti.math.pi

# angle = 0.0
xaxis = [1.0, 0.0, 0.0]
yaxis = [0.0, 1.0, 0.0]
zaxis = [0.0, 0.0, 1.0]

# rotate position
@ti.kernel
def rotate(axis: vec3, angle: ti.f32, vec: ti.template()) -> vec3:
    # axis: vec3
    # angle: float(radian)
    rotatemat = ti.math.rot_by_axis(axis, angle)    
    translatemat = ti.math.translate(-center[0], -center[1], -center[2])
    invtranslatemat = ti.math.translate(center[0], center[1], center[2])
    newvec = invtranslatemat @ rotatemat @ translatemat @ vec
    return ti.Vector([newvec[0], newvec[1], newvec[2]])

# rotate vector
@ti.kernel
def rotatevector(axis: vec3, angle: ti.f32, vec: ti.template()) -> vec3:
    # axis: vec3
    # angle: float(radian)
    rotatemat = ti.math.rot_by_axis(axis, angle)    
    newvec = rotatemat @ vec
    return ti.Vector([newvec[0], newvec[1], newvec[2]])


tmp = xaxis 
tmp.append(0.0)
xaxis = rotatevector(rotateaxis, angle, ti.Vector(tmp))
tmp = yaxis 
tmp.append(0.0)
yaxis = rotatevector(rotateaxis, angle, ti.Vector(tmp))
tmp = zaxis 
tmp.append(0.0)
zaxis = rotatevector(rotateaxis, angle, ti.Vector(tmp))

for i in range(8):
    # vertices[i] = ti.Vector(vertices_list[i])
    tmp = vertices_list[i]
    tmp.append(1.0)
    vertices[i] = rotate(rotateaxis, angle, ti.Vector(tmp))

indices = ti.field(int, 36)
indices_list = [0, 2, 6, 
                6, 4, 0,
                4, 5, 1,
                1, 0, 4,
                5, 7, 3,
                3, 1, 5,
                6, 2, 3,
                3, 7, 6,
                6, 7, 5, 
                5, 4, 6,
                1, 3, 2,
                2, 0, 1]
for i in range(36):
    indices[i] = indices_list[i]

@ti.func
def detect_and_correct(coor, v):
    dis = (coor / n_grid) - center
    xdis = ti.math.dot(dis, xaxis)
    ydis = ti.math.dot(dis, yaxis)
    zdis = ti.math.dot(dis, zaxis)
    # collision detection
    if -r <= xdis <= r and -r <= ydis <= r and -r <= zdis <= r:
        vx = ti.math.dot(v, xaxis)
        vy = ti.math.dot(v, yaxis)
        vz = ti.math.dot(v, zaxis)

        if -r <= xdis <= 0 and vx > 0:
            vx = 0
        elif 0 <= xdis <= r and vx < 0:
            vx = 0
            
        if -r <= ydis <= 0 and vy > 0:
            vy = 0
        elif 0 <= xdis <= r and vy < 0:
            vy = 0
        
        if -r <= zdis <= 0 and vz > 0:
            vz = 0
        elif 0 <= zdis <= r and vz < 0:
            vz = 0
        v = vx * xaxis + vy * yaxis  + vz * zaxis

    return v


@ti.kernel
def MLSMPM_substep(g_x: float, g_y: float, g_z: float):
    # set to zero
    for I in ti.grouped(F_grid_m):
        F_grid_v[I] = ti.zero(F_grid_v[I])
        F_grid_m[I] = 0
        F_grid_f[I] = ti.zero(F_grid_f[I])
    ti.loop_config(block_dim=n_grid)
    # P2G
    for p in F_x:
        if F_used[p] == 0:
            continue
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        F_dg[p] = (ti.Matrix.identity(float, 3) +
                  dt * F_C[p]) @ F_dg[p]  # deformation gradient update
        # Hardening coefficient: snow gets harder when compressed
        h = ti.exp(hardening * (1.0 - F_Jp[p]))
        mu, la = mu_0 * h, lambda_0 * h

        U, sig, V = ti.svd(F_dg[p])
        J = 1.0
        for d in ti.static(range(3)):
            new_sig = sig[d, d]
            new_sig = ti.min(ti.max(sig[d, d], 1-critical_compression), 1+critical_stretch)
            F_Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        F_dg[p] = U @ sig @ V.transpose()
        stress = 2 * mu * (F_dg[p] - U @ V.transpose()) @ F_dg[p].transpose(
        ) + ti.Matrix.identity(float, 3) * la * J * (J - 1)
        stress = (-dt * p_vol * 4) * stress / dx**2
        affine = stress + p_mass * F_C[p]

        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            F_grid_v[base +
                    offset] += weight * (p_mass * F_v[p] + affine @ dpos)
            F_grid_m[base + offset] += weight * p_mass
    for I in ti.grouped(F_grid_m):
        if F_grid_m[I] > 0:
            F_grid_v[I] /= F_grid_m[I]
        F_grid_v[I] += dt * ti.Vector([g_x, g_y, g_z])
        F_grid_v[I] = detect_and_correct(I, F_grid_v[I])

        cond = (I < bound) & (F_grid_v[I] < 0) | \
               (I > n_grid - bound) & (F_grid_v[I] > 0)
        F_grid_v[I] = ti.select(cond, 0, F_grid_v[I])
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        if F_used[p] == 0:
            continue
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.zero(F_v[p])
        new_C = ti.zero(F_C[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = F_grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        F_v[p] = new_v
        F_x[p] += dt * F_v[p]
        F_C[p] = new_C

@ti.kernel
def MPM_substep(g_x: float, g_y: float, g_z: float):
    # set to zero
    for I in ti.grouped(F_grid_m):
        F_grid_v[I] = ti.zero(F_grid_v[I])
        F_grid_m[I] = 0
        F_grid_f[I] = ti.zero(F_grid_f[I])
    ti.loop_config(block_dim=n_grid)
    # P2G
    for p in F_x:
        if F_used[p] == 0:
            continue
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        w_grad = [fx-1.5, -2*(fx-1), fx-3.5] # Bspline gradient
        # Hardening coefficient: snow gets harder when compressed
        h = ti.exp(hardening * (1.0 - F_Jp[p]))
        mu, la = mu_0 * h, lambda_0 * h

        U, sig, V = ti.svd(F_dg[p])
        J = 1.0
        for d in ti.static(range(3)):
            J *= sig[d, d]
        stress = 2 * mu * (F_dg[p] - U @ V.transpose()) @ F_dg[p].transpose(
        ) + ti.Matrix.identity(float, 3) * la * J * (J - 1)
        stress = -p_vol * stress

        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            weight_grad = vec3(w_grad[offset[0]][0] * w[offset[1]][1] * w[offset[2]][2], w[offset[0]][0] * w_grad[offset[1]][1] * w[offset[2]][2], w[offset[0]][0] * w[offset[1]][1] * w_grad[offset[2]][2])
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            F_grid_v[base + offset] += weight * p_mass * F_v[p]
            F_grid_m[base + offset] += weight * p_mass
            F_grid_f[base + offset] += stress @ weight_grad
    for I in ti.grouped(F_grid_m):
        if F_grid_m[I] > 0:
            F_grid_v[I] /= F_grid_m[I]
            F_grid_v[I] += dt * (F_grid_f[I] / F_grid_m[I])
        F_grid_v[I] += dt * ti.Vector([g_x, g_y, g_z])
        F_grid_v[I] = detect_and_correct(I, F_grid_v[I])

        cond = (I < bound) & (F_grid_v[I] < 0) | \
               (I > n_grid - bound) & (F_grid_v[I] > 0)
        F_grid_v[I] = ti.select(cond, 0, F_grid_v[I])
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        if F_used[p] == 0:
            continue
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        w_grad = [fx-1.5, -2*(fx-1), fx-3.5] # Bspline gradient
        new_v = ti.zero(F_v[p])
        new_C = ti.zero(F_C[p])
        incre = ti.zero(F_dg[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            weight_grad = vec3(w_grad[offset[0]][0] * w[offset[1]][1] * w[offset[2]][2], w[offset[0]][0] * w_grad[offset[1]][1] * w[offset[2]][2], w[offset[0]][0] * w[offset[1]][1] * w_grad[offset[2]][2])
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = F_grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
            incre += (dt * g_v).outer_product(weight_grad)
        F_v[p] = new_v
        F_dg[p] += incre @ F_dg[p]
        F_x[p] += dt * F_v[p]
        F_C[p] = new_C
        U, sig, V = ti.svd(F_dg[p])
        J = 1.0
        for d in ti.static(range(3)):
            new_sig = sig[d, d]
            new_sig = ti.min(ti.max(sig[d, d], 1-critical_compression), 1+critical_stretch)
            F_Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        F_dg[p] = U @ sig @ V.transpose()

class CubeVolume:
    def __init__(self, minimum, size, material):
        self.minimum = minimum
        self.size = size
        self.volume = self.size.x * self.size.y * self.size.z
        self.material = material


@ti.kernel
def init_cube_vol(first_par: int, last_par: int, x_begin: float,
                  y_begin: float, z_begin: float, x_size: float, y_size: float,
                  z_size: float, material: int):
    for i in range(first_par, last_par):
        F_x[i] = ti.Vector([ti.random() for i in range(dim)]) * ti.Vector(
            [x_size, y_size, z_size]) + ti.Vector([x_begin, y_begin, z_begin])
        F_Jp[i] = 1
        F_dg[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        F_v[i] = ti.Vector([0.0, 0.0, 0.0])
        F_materials[i] = material
        F_colors_random[i] = ti.Vector(
            [ti.random(), ti.random(),
             ti.random(), ti.random()])
        F_used[i] = 1


@ti.kernel
def set_all_unused():
    for p in F_used:
        F_used[p] = 0
        # basically throw them away so they aren't rendered
        F_x[p] = ti.Vector([533799.0, 533799.0, 533799.0])
        F_dg[p] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        F_C[p] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        F_v[p] = ti.Vector([0.0, 0.0, 0.0])
        F_Jp[p] = 1


def init_vols(vols):
    set_all_unused()
    total_vol = 0
    for v in vols:
        total_vol += v.volume

    next_p = 0
    for i, v in enumerate(vols):
        v = vols[i]
        if isinstance(v, CubeVolume):
            par_count = int(v.volume / total_vol * n_particles)
            if i == len(
                    vols
            ) - 1:  # this is the last volume, so use all remaining particles
                par_count = n_particles - next_p
            init_cube_vol(next_p, next_p + par_count, *v.minimum, *v.size,
                          v.material)
            next_p += par_count
        else:
            raise Exception("???")


@ti.kernel
def set_color_by_material(mat_color: ti.types.ndarray()):
    for i in range(n_particles):
        mat = F_materials[i]
        F_colors[i] = ti.Vector(
            [mat_color[mat, 0], mat_color[mat, 1], mat_color[mat, 2], 1.0])


print("Loading presets...this might take a minute")

presets = [[
               CubeVolume(ti.Vector([0.35, 0.7, 0.0]),
                          ti.Vector([0.2, 0.2, 0.2]), SNOW)
           ]]
preset_names = [
    "Customize"
]

curr_preset_id = 0

paused = False

use_random_colors = False
particles_radius = 0.02

material_colors = [(0.1, 0.6, 0.9), (0.93, 0.33, 0.23), (1.0, 1.0, 1.0)]


def init():
    global paused
    init_vols(presets[curr_preset_id])


init()

res = (1080, 720)
window = ti.ui.Window("Real MPM 3D", res, vsync=True)

canvas = window.get_canvas()
gui = window.get_gui()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(0.5, 1.0, 1.95)
camera.lookat(0.5, 0.3, 0.5)
camera.fov(55)


def show_options():
    global use_random_colors
    global paused
    global particles_radius
    global curr_preset_id

    with gui.sub_window("Gravity", 0.05, 0.3, 0.2, 0.1) as w:
        GRAVITY[0] = w.slider_float("x", GRAVITY[0], -10, 10)
        GRAVITY[1] = w.slider_float("y", GRAVITY[1], -10, 10)
        GRAVITY[2] = w.slider_float("z", GRAVITY[2], -10, 10)

    with gui.sub_window("Options", 0.05, 0.45, 0.2, 0.4) as w:
        use_random_colors = w.checkbox("use_random_colors", use_random_colors)
        if not use_random_colors:
            material_colors[WATER] = w.color_edit_3("water color",
                                                    material_colors[WATER])
            material_colors[SNOW] = w.color_edit_3("snow color",
                                                   material_colors[SNOW])
            material_colors[JELLY] = w.color_edit_3("jelly color",
                                                    material_colors[JELLY])
            set_color_by_material(np.array(material_colors, dtype=np.float32))
        particles_radius = w.slider_float("particles radius ",
                                          particles_radius, 0, 0.1)
        if w.button("restart"):
            init()
        if paused:
            if w.button("Continue"):
                paused = False
        else:
            if w.button("Pause"):
                paused = True


def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light((0, 0, 0))
    scene.mesh(vertices, color=(1.0, 1.0, 1.0), indices=indices)

    colors_used = F_colors_random if use_random_colors else F_colors
    scene.particles(F_x, per_vertex_color=colors_used, radius=particles_radius)

    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))

    canvas.scene(scene)


def main():
    frame_id = 0
    series_prefix = mpm_directory + "/" + ply_directory + "/ex.ply"
    # series_prefix = "MPM_ply/reference_ply/ex.ply"
    while window.running:
        frame_id += 1
        frame_id = frame_id % 1024

        if not paused:
            for _ in range(steps):
                if MPM_TYPE == MPM:
                    MPM_substep(*GRAVITY)
                elif MPM_TYPE == MLS_MPM:
                    MLSMPM_substep(*GRAVITY)
                    

        render()
        show_options()
        window.show()
        np_pos = np.reshape(F_x.to_numpy(), (n_particles, 3))
        np_rgba = np.reshape(F_colors.to_numpy(), (n_particles, 4))
        # create a PLYWriter
        writer = ti.tools.PLYWriter(num_vertices=n_particles)
        writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
        writer.add_vertex_rgba(np_rgba[:, 0], np_rgba[:, 1], np_rgba[:, 2],
                            np_rgba[:, 3])
        writer.export_frame(frame_id, series_prefix)


if __name__ == '__main__':
    main()