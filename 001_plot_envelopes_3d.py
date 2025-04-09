import xtrack as xt
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

env = xt.load_madx_lattice(file='EYETS 2024-2025.seq', reverse_lines=['lhcb2'])

env.lhcb1.particle_ref = xt.Particles(p0c=6.8e12)
env.lhcb2.particle_ref = xt.Particles(p0c=6.8e12)

env.vars.load_madx('ats_30cm.madx')
# env['on_sep5'] = 0

tw1 = env.lhcb1.twiss4d()
tw2 = env.lhcb2.twiss4d(reverse=True)

sv1 = env.lhcb1.survey()
sv2 = env.lhcb2.survey().reverse()


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


def ellipse(rxy, rz, beam_xy, beam_z, x, y, z, theta):
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
    points_xz = Rotation.from_euler('z', theta).apply(points_xz)
    return points_xz + np.tile([x, y, z], (len(ts), 1))


def mesh_from_polygons(pts, close=True):
    num_ellipses, points_per_ellipse, _ = pts.shape
    vertices = pts.reshape(-1, 3)
    num_faces = points_per_ellipse * (num_ellipses - 1) - 1

    faces = np.hstack([
        [4, i, i + 1, points_per_ellipse + i + 1, points_per_ellipse + i]
        for i in range(num_faces)
        if close or (i % points_per_ellipse != points_per_ellipse - 1)
    ])

    surface = pv.PolyData(vertices, faces)
    return surface


def plot_beam_size(ax, twiss, survey, color, element_around, section_length):
    s_around = twiss.rows[element_around].s[0]
    s_start, s_end = s_around - section_length / 2, s_around + section_length / 2

    sv = survey.rows[s_start:s_end:'s']
    tw = twiss.rows[s_start:s_end:'s']

    s, x, sigx, y, sigy, sx, sy, sz, theta = compute_beam_size(sv, tw)

    pts = np.array([
        ellipse(sigx[i], sigy[i], x[i], y[i], sx[i], sz[i], sy[i], theta[i])
        for i in range(len(sigx))
    ])

    # Plot the envelopes
    surface = mesh_from_polygons(pts)
    ax.add_mesh(surface, color=color, opacity=0.5, show_edges=True)

    # Plot the closed orbit
    center = np.column_stack([
        sx + np.cos(theta) * x,
        sz + np.sin(theta) * x,
        sy + y,
    ])
    spline = pv.Spline(center)
    ax.add_mesh(spline, color=color, line_width=5)


def make_screen(x_min, x_max, y_min, y_max, beam_xy, beam_z, x, y, z, theta):
    """Make a beam screen shape.

    Make a rectangle at ``(x, y, z)``, rotated around z-axis by the angle ``theta``,
    spanning from x_min and y_min to x_max and y_max. The axes are the traditional
    (matplotlib) axes.

    Parameters
    ----------
    x_min : float
        Minimum x extent.
    x_max : float
        Maximum x extent.
    y_min : float
        Minimum y extent.
    y_max : float
        Maximum y extent.
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
    points_xz = np.array([
        (0, 0, y_min),
        (x_min, 0, y_min),
        (x_min, 0, y_max),
        (0, 0, y_max),
    ])
    points_xz = Rotation.from_euler('z', theta).apply(points_xz)
    return points_xz + np.tile([x, y, z], (len(points_xz), 1))


def plot_apertures(ax, apertures, twiss, survey, name_from, name_until):
    sv = survey.rows[name_from:name_until]
    ap = apertures.rows[name_from:name_until]
    tw = twiss.rows[name_from:name_until]

    s, x, sigx, y, sigy, sx, sy, sz, theta = compute_beam_size(sv, tw)

    x_min, x_max = ap.x_aper_low, ap.x_aper_high
    y_min, y_max = ap.y_aper_low, ap.y_aper_high

    pts = np.array([
        make_screen(x_min[i], x_max[i], y_min[i], y_max[i], x[i], y[i], sx[i], sz[i], sy[i], theta[i])
        for i in range(len(x_min))
    ])

    # Plot the screen
    surface = mesh_from_polygons(pts, close=False)
    ax.add_mesh(surface, color='k', opacity=0.5, show_edges=True)


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

ax.set_scale(xscale=3e2, zscale=3e2)
ax.show_bounds(
    show_xaxis=False,
    show_yaxis=True,
    show_zaxis=False,
    show_xlabels=False,
    show_ylabels=True,
    show_zlabels=False,
    ytitle='Z [m]',
    location='origin',
)
title = ax.add_title(f'LHC Beam Envelopes at CMS (3σ, β*=30cm)')
title_text_prop = title.GetTextProperty()
title_text_prop.SetFontFamily(4)
title_text_prop.SetFontFile('/Users/szymonlopaciuk/Library/Fonts/DejaVuSans.ttf')

plot_beam_size(ax, tw1, sv1, color='b', element_around='ip5', section_length=130)
plot_beam_size(ax, tw2, sv2, color='r', element_around='ip5', section_length=130)

# Plot beam screen
# ================

lhcb1_aper = xt.Line.from_json('lhcb1_aper.json')
tw1 = lhcb1_aper.twiss4d()

element_around = 'ip5'
section_length = 130

s_around = sv1.rows[element_around].s[0]
s_start, s_end = s_around - section_length / 2, s_around + section_length / 2

name_from = tw1.rows[s_start:s_end:'s'].name[0]
name_until = tw1.rows[s_start:s_end:'s'].name[-1]

aper = lhcb1_aper.select(name_from, name_until).get_aperture_table()
sv1 = lhcb1_aper.survey()

plot_apertures(ax, aper, tw1, sv1, name_from, name_until)

ax.show()
