# Lorenz

Lorenz provides simple utilities to generate and plot Lorenz system trajectories (the classic "butterfly" attractor).

## Install

```text
pip install lorenz
```

## Quick start

```python
from lorenz import LorenzButterfly

lorenz = LorenzButterfly()
ax = lorenz.plot()
```

## API

### LorenzButterfly

Generate Lorenz system data and plot trajectories.

```python
from lorenz import LorenzButterfly

lorenz = LorenzButterfly(sigma=10.0, rho=28.0, beta=8.0 / 3.0)

# Generate data as a pandas DataFrame
frame = lorenz.generate_data(steps=5000, dt=0.01)

# Plot the trajectory (matplotlib 3D axes)
ax = lorenz.plot(steps=5000, dt=0.01)
```

**Parameters**
- `sigma`: Prandtl number controlling horizontal convection strength
- `rho`: Rayleigh number controlling temperature gradient/chaos level
- `beta`: geometry factor (often `8/3` for the classic system)
- `dt`: integration time step
- `steps`: number of integration steps
- `initial`: starting point `(x, y, z)`

**Methods**
- `generate_data(...)` returns a `pandas.DataFrame` with columns `x`, `y`, `z`
- `plot(...)` returns a `matplotlib.axes.Axes` instance
