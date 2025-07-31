import xtrack as xt
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

# Options
# =======
element_around = 'ip1'
colour = 'gray'
section_length = 140
ips = {
    'ip1': 'Atlas',
    'ip2': 'Alice',
    'ip5': 'CMS',
    'ip8': 'LHCb',
}

# Prepare the lattice and plot the beam
# =====================================
env = xt.load_madx_lattice(file='EYETS 2024-2025.seq', reverse_lines=['lhcb2'])

env.lhcb1.particle_ref = xt.Particles(p0c=6.8e12)
env.lhcb2.particle_ref = xt.Particles(p0c=6.8e12)

env.vars.load_madx('ats_30cm.madx')
env['on_sep1'] = 0
env['on_sep2'] = 0
env['on_sep5'] = 0
env['on_sep8'] = 0

# def cut_around(line, where, length, resolution):
#     s_around = line.get_table().rows[where].s[0]
#     s_start, s_end = s_around - length / 2, s_around + length / 2
#     cuts = np.linspace(s_start, s_end, resolution) % line.get_length()
#     line.cut_at_s(cuts, s_tol=0.01)
#
# cut_around(env.lhcb1, element_around, section_length, 200)
# cut_around(env.lhcb2, element_around, section_length, 200)

tw1 = env.lhcb1.twiss4d()
tw2 = env.lhcb2.twiss4d(reverse=True)

sv1 = env.lhcb1.survey()
sv2 = env.lhcb2.survey().reverse()


@np.vectorize
def clip_large_s(s):
    if s > env.lhcb1.get_length() / 2:
        return s - env.lhcb1.get_length()
    return s


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


sv1 = table_wrap_around(sv1, element_around, section_length)
tw1 = table_wrap_around(tw1, element_around, section_length)
sv2 = table_wrap_around(sv2, element_around, section_length)
tw2 = table_wrap_around(tw2, element_around, section_length)


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
    # centre = Rotation.from_euler('z', theta).apply(centre)
    # centre[0] += s
    centre = [0, clip_large_s(s), 0]

    return points_xz + np.tile(centre, (len(ts), 1))


def mesh_from_polygons(pts, close=True):
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
    ax.add_mesh(surface, color=color, opacity=0.5, show_edges=True)

    # Plot the closed orbit
    center = np.column_stack(
        [
            x[:min_len] * scale,
            clip_large_s(s),
            y[:min_len] * scale,
        ]
    )
    spline = pv.Spline(center)
    ax.add_mesh(spline, color=color, line_width=5)


def make_screen(xs, ys, x, y, z, theta, s):
    """Make a beam screen shape.

    Make a polygon at ``(x, y, z)``, rotated around z-axis by the angle ``theta``,
    consisting of points (xs, ys). The axes are the traditional (matplotlib) axes.

    Parameters
    ----------
    xs : array of float
        x-coordinates of the polygon corners.
    ys : array of float
        y-coordinates of the polygon corners.
    x : float
        Centre of the ellipse in x.
    y : float
        Centre of the ellipse in y.
    z : float
        Centre of the ellipse in z.
    theta : float
        Angle of rotation around the z-axis.
    """
    points_xz = np.column_stack([xs, np.zeros_like(xs), ys])
    #points_xz = Rotation.from_euler('z', -theta).apply(points_xz)

    # centre = [x, y, z]
    # centre = Rotation.from_euler('z', theta).apply(centre)
    centre = [0, clip_large_s(s), 0]

    return points_xz + np.tile(centre, (len(points_xz), 1))


def plot_apertures(ax, aperture, survey, close=True, scale=1e3):
    sx = survey.X
    sy = survey.Y
    sz = survey.Z
    s = survey.s
    theta = survey.theta

    xs, ys = aperture.polygon_x_discrete, aperture.polygon_y_discrete

    aper_indices = np.where(aperture.aperture_mask)[0]
    pts = np.array([
        make_screen(xs[i] * scale, ys[i] * scale, sx[i], sz[i], sy[i], theta[i], s[i])
        for i in aper_indices
    ])

    surface = mesh_from_polygons(pts, close=close)
    ax.add_mesh(surface, color=colour, edge_color='k', opacity=0.3, show_edges=True)


def make_rectangular_screen(x_min, x_max, y_min, y_max, x, y, z, theta, close=True):
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
    if not close:
        x_max = 0

    points_xz = np.array([
        (x_max, 0, y_min),
        (x_min, 0, y_min),
        (x_min, 0, y_max),
        (x_max, 0, y_max),
    ])
    points_xz = Rotation.from_euler('z', -theta).apply(points_xz)
    return points_xz + np.tile([x, y, z], (len(points_xz), 1))


def plot_rectangular_apertures(ax, aperture, survey, close=True, scale=1e3):
    sx = survey.X
    sy = survey.Y
    sz = survey.Z
    theta = survey.theta

    x_min, x_max = aperture.x_aper_low * scale, aperture.x_aper_high * scale
    y_min, y_max = aperture.y_aper_low * scale, aperture.y_aper_high * scale

    pts = np.array([
        make_rectangular_screen(x_min[i], x_max[i], y_min[i], y_max[i], sx[i], sz[i], sy[i], theta[i], close)
        for i in range(len(x_min))
    ])

    # Plot the screen
    surface = mesh_from_polygons(pts, close)
    ax.add_mesh(surface, color='g', opacity=0.5, show_edges=True)


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
    xscale=3,
    yscale=1,
    zscale=3,
)
scale = 100
ax.show_bounds(
    show_xaxis=True,
    show_yaxis=True,
    show_zaxis=True,
    show_xlabels=True,
    show_ylabels=True,
    show_zlabels=True,
    xtitle='X [mm]',
    ytitle='Z [m]',
    ztitle='Y [mm]',
    location='outer',
)
experiment = f' ({ips[element_around]})' if element_around in ips else ''
title = ax.add_title(f'LHC Beam Envelopes at {element_around.upper()}{experiment} (3σ, β*=30cm)')
title_text_prop = title.GetTextProperty()
title_text_prop.SetFontFamily(4)
title_text_prop.SetFontFile('/Users/szymonlopaciuk/Library/Fonts/DejaVuSans.ttf')

plot_beam_size(ax, tw1, sv1, color='b', scale=scale)
plot_beam_size(ax, tw2, sv2, color='r', scale=scale)

# Plot beam screen
# ================

lhcb1_aper = xt.Line.from_json('lhcb1_aper.json')
sv_aper = lhcb1_aper.survey()
sv_aper = table_wrap_around(sv_aper, element_around, section_length)

name_from = sv_aper.name[0]
name_until = sv_aper.name[-1]

aper_section = lhcb1_aper.cycle(name_from).select(name_from, name_until)
aper = aper_section.get_aperture_table(option='poly')
aper_sq = aper_section.get_aperture_table(option='extent')

plot_apertures(ax, aper, sv_aper, scale=scale, close=False)
# plot_rectangular_apertures(ax, aper_sq, sv_aper, name_from, name_until)

ax.show()
