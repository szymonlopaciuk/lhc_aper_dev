import xtrack as xt

env = xt.load_madx_lattice(file='EYETS 2024-2025.seq', reverse_lines=['lhcb2'])

svb1 = env.lhcb1.survey()
svb2 = env.lhcb2.survey().reverse()

# Load aperture markers in a dummy sequence
from cpymad.madx import Madx
mad = Madx()
mad.input(f'''
LHCB1 : SEQUENCE, refer = centre,    L = 28000;
ip1: marker, at = {svb1['s', 'ip1']:.6f};
ip2: marker, at = {svb1['s', 'ip2']:.6f};
ip3: marker, at = {svb1['s', 'ip3']:.6f};
ip4: marker, at = {svb1['s', 'ip4']:.6f};
ip5: marker, at = {svb1['s', 'ip5']:.6f};
ip6: marker, at = {svb1['s', 'ip6']:.6f};
ip7: marker, at = {svb1['s', 'ip7']:.6f};
ip8: marker, at = {svb1['s', 'ip8']:.6f};
ip1.l1: marker, at = {svb1['s', 'ip1.l1']:.6f};
endsequence;
LHCB2 : SEQUENCE, refer = centre,    L = 28000;
ip1: marker, at = {svb2['s', 'ip1']:.6f};
ip2: marker, at = {svb2['s', 'ip2']:.6f};
ip3: marker, at = {svb2['s', 'ip3']:.6f};
ip4: marker, at = {svb2['s', 'ip4']:.6f};
ip5: marker, at = {svb2['s', 'ip5']:.6f};
ip6: marker, at = {svb2['s', 'ip6']:.6f};
ip7: marker, at = {svb2['s', 'ip7']:.6f};
ip8: marker, at = {svb2['s', 'ip8']:.6f};
ip1.l1: marker, at = {svb2['s', 'ip1.l1']:.6f};
endsequence;
''')
mad.call('APERTURE_EYETS 2024-2025.seq')
mad.input('''
    beam, particle=proton, energy=7000, sequence=lhcb1;
    beam, particle=proton, energy=7000, sequence=lhcb2, bv=-1;
    ''')

mad.use('LHCB1')
# mad.use('LHCB2')

line_b1_aper = xt.Line.from_madx_sequence(mad.sequence.LHCB1, install_apertures=True)
# line_b2_aper = xt.Line.from_madx_sequence(mad.sequence.LHCB2, install_apertures=True)


line_aper = line_b1_aper.copy()
line = env.lhcb1

# Identify the aperture markers
tt_aper = line_aper.get_table().rows['.*_aper']

# Prepare insertions
env = line.env
insertions = []
for nn in tt_aper.name:

    elem = line_aper.get(nn).copy()

    env.elements[nn] = elem
    insertions.append(env.place(nn, at=tt_aper['s', nn]))

# Insert the apertures into the line
line.insert(insertions, s_tol=1e-6)

line.vars.load_madx('ats_30cm.madx')
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, p0c=6800e9)

tw = line.twiss4d()

aper = line.get_aperture_table()

import matplotlib.pyplot as plt
plt.close('all')
tw.plot(lattice_only=True)
plt.plot(aper.s, aper.x_aper_low, 'k-')
plt.plot(aper.s, aper.x_aper_high, 'k-')
plt.plot(aper.s, aper.x_aper_low_discrete, '.k')
plt.plot(aper.s, aper.x_aper_high_discrete, '.k')
plt.plot(aper.s, aper.y_aper_low, 'r-')
plt.plot(aper.s, aper.y_aper_high, 'r-')
plt.plot(aper.s, aper.y_aper_low_discrete, '.r')
plt.plot(aper.s, aper.y_aper_high_discrete, '.r')
plt.show()

line.to_json('lhcb1_aper.json')
