import xtrack as xt
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

# Options
# =======
colour = 'gray'

# Prepare the lattice and plot the beam
# =====================================
pi = np.pi
lbend = 3

# Create an environment
env = xt.Environment()

# Build a simple ring
k1 = 1
line = env.new_line(components=[
    env.new('mqf.1', xt.Quadrupole, length=0.3, k1=k1),
    env.new('d1.1',  xt.Drift, length=5),

    env.new('mqd.1', xt.Quadrupole, length=0.3, k1=-k1),
    env.new('d3.1',  xt.Drift, length=5),

    env.new('mqf.2', xt.Quadrupole, length=0.3, k1=k1),
    env.new('d1.2',  xt.Drift, length=5),

    env.new('mqd.2', xt.Quadrupole, length=0.3, k1=-k1),
    env.new('d3.2',  xt.Drift, length=5),
])
line.cut_at_s(np.linspace(0, line.get_length(), 50))

# Define reference particle
line.particle_ref = xt.Particles(p0c=1e9, mass0=xt.PROTON_MASS_EV)

# def cut_around(line, where, length, resolution):
#     s_around = line.get_table().rows[where].s[0]
#     s_start, s_end = s_around - length / 2, s_around + length / 2
#     cuts = np.linspace(s_start, s_end, resolution) % line.get_length()
#     line.cut_at_s(cuts, s_tol=0.01)
#
# cut_around(env.lhcb1, element_around, section_length, 200)
# cut_around(env.lhcb2, element_around, section_length, 200)

tw1 = line.twiss4d()
sv1 = line.survey()

def table_wrap_around(tb, element_around, section_length):
    s_around = tb.rows[element_around].s[0]
    length = tb.s[-1]
    s_start = (s_around - section_length / 2) % length
    s_end = (s_around + section_length / 2) % length

    if s_start < s_end:
        return tb.rows[s_start:s_end:'s']

    tb_end = tb.rows[s_start:length:'s']
    tb_start = tb.rows[0:s_end:'s']
    return tb_end + tb_start


def compute_beam_size(survey, twiss):
    sx = survey.X
    sy = survey.Y
    sz = survey.Z
    theta = survey.theta
    s = twiss.s
    x = twiss.x
    y = twiss.y
    bx = twiss.betx
    by = twiss.bety
    dx = twiss.dx
    dy = twiss.dy
    nemitt_x = 2.5e-6
    nemitt_y = 2.5e-6
    gamma0 = twiss.gamma0
    n_sigmas = 3 # 13.
    sigma_delta = 8e-4

    sigx = n_sigmas * np.sqrt(nemitt_x / gamma0 * bx) + abs(dx) * sigma_delta
    sigy = n_sigmas * np.sqrt(nemitt_y / gamma0 * by) + abs(dy) * sigma_delta

    return s, x, sigx, y, sigy, sx, sy, sz, theta


def ellipse(rxy, rz, beam_xy, beam_z, x, y, z, theta, s):
    """Make a 3D ellipse.

    Make a 3D ellipse centred at ``(x, y, z)``, with radii ``rx`` and ``rz``, and
    rotated around z-axis by the angle ``theta``. The axes are the traditional
    (matplotlib) axes.

    Parameters
    ----------
    rxy : float
        Radius in the xy-plane.
    rz : float
        z-axis radius.
    beam_xy : float
        Horizontal displacement of the centre before rotation, i.e. along theta.
    beam_z : float
        Vertical displacement of the centre before rotation.
    x : float
        Centre of the ellipse in x.
    y : float
        Centre of the ellipse in y.
    z : float
        Centre of the ellipse in z.
    theta : float
        Angle of rotation around the z-axis.
    """
    ts = np.linspace(0, 2 * np.pi, 20)
    points_xz = np.array([
        (rxy * np.cos(t) + beam_xy, 0, rz * np.sin(t) + beam_z) for t in ts]
    )
    # points_xz = Rotation.from_euler('z', theta).apply(points_xz)

    # centre = [x, y, z]
    centre = [0, s, 0]

    return points_xz + np.tile(centre, (len(ts), 1))


def mesh_from_polygons(pts, close=False):
    num_polys, points_per_poly, dim = pts.shape
    assert dim == 3, "Points must be 3D"
    vertices = pts.reshape(-1, 3)
    num_faces = points_per_poly * (num_polys - 1) - 1

    faces = np.hstack([
        [4, i, i + 1, points_per_poly + i + 1, points_per_poly + i]
        for i in range(num_faces)
        if close or (i % points_per_poly != points_per_poly - 1)
    ])

    surface = pv.PolyData(vertices, faces)
    return surface


def plot_beam_size(ax, twiss, survey, color, scale=1e3):
    s, x, sigx, y, sigy, sx, sy, sz, theta = compute_beam_size(survey, twiss)
    min_len = min(len(x), len(theta))  # these can be off by one due to numerical precision??

    pts = np.array([
        ellipse(sigx[i] * scale, sigy[i] * scale, x[i] * scale, y[i] * scale, sx[i], sz[i], sy[i], -theta[i], s[i])
        for i in range(min_len)
    ])

    # Plot the envelopes
    surface = mesh_from_polygons(pts)
    ax.add_mesh(surface, color=color, opacity=1, show_edges=True, lighting=True)

    # Plot the closed orbit
    # center = np.column_stack([
    #     x[:min_len] * scale,
    #     s,
    #     y[:min_len] * scale,
    # ])
    # spline = pv.Spline(center)
    # ax.add_mesh(spline, color=color, line_width=5)


# Plot the plot
# =============
ax = pv.Plotter()

ax.add_axes(
    line_width=5,
    cone_radius=0.6,
    shaft_length=0.7,
    tip_length=0.3,
    ambient=0.5,
    label_size=(0.4, 0.16),
    xlabel='X',
    ylabel='Z',
    zlabel='Y',
)
ax.set_scale(
    xscale=1,
    yscale=1,
    zscale=1,
)
# ax.set_scale(
#     xscale=3,
#     yscale=1,
#     zscale=3,
# )
scale = 100
ax.show_bounds(
    show_xaxis=True,
    show_yaxis=True,
    show_zaxis=True,
    show_xlabels=True,
    show_ylabels=True,
    show_zlabels=True,
    xtitle='X [cm]',
    ytitle='Z [m]',
    ztitle='Y [cm]',
    location='outer',
)
title = ax.add_title(f'Toy 2×FODO Beam Envelope (3σ, 1GeV p+, k1=1)')
title_text_prop = title.GetTextProperty()
title_text_prop.SetFontFamily(4)
title_text_prop.SetFontFile('/Users/szymonlopaciuk/Library/Fonts/DejaVuSans.ttf')

plot_beam_size(ax, tw1, sv1, color=colour, scale=scale)

ax.show()
