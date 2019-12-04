def heatpotential(t_data, offset, tc0, cc):
    t_data = (t_data+offset)/tc0
    return t_data*(cc+0.5*t_data)/(1.0+t_data)
