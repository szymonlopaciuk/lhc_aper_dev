import xtrack as xt
import numpy as np
import matplotlib.pyplot as plt

env = xt.Environment()

line = xt.Line(elements=[
        xt.LimitEllipse(a=0.03, b=0.05),
        xt.Drift(length=5),
        xt.Drift(length=5),
        xt.LimitEllipse(a=0.05, b=0.03),
        xt.Drift(length=5),
        xt.Drift(length=5),
        xt.LimitRect(min_x=-0.03, max_x=0.04, min_y=-0.05, max_y=0.06),
    ],
)
line.particle_ref = xt.Particles(p0c=6.8e12)

aper = line.get_aperture_table(option='poly')
aper_sq = line.get_aperture_table()

for element in line.element_names:
    plt.figure()

    plt.title(f'Aperture for {element} : {type(line.element_dict[element]).__name__}, s = {aper.rows[element].s[0]}')
    i = np.where(aper.name == element)[0][0] + 1

    plt.plot(aper_sq.x_aper_low[i], aper_sq.y_aper_low[i], 'o', c='g')
    plt.plot(aper_sq.x_aper_high[i], aper_sq.y_aper_high[i], 'o', c='g')
    plt.axis('equal')

    xs = aper.polygon_x[i]
    xs = np.concatenate((xs, [xs[0]]))
    ys = aper.polygon_y[i]
    ys = np.concatenate((ys, [ys[0]]))

    plt.plot(xs, ys, '.', c='k')

    plt.scatter(aper.all_x[i], aper.all_y[i], c=np.where(aper.all_state[i] > 0, 'b', 'r'))

    plt.show()
