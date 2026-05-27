from matplotlib import pyplot as plt

import indago
import numpy as np
from copy import deepcopy

dims = 15
real: indago.VariableDictType = {f'var{i}': (indago.VariableType.Real, 0, 360) for i in range(0, dims)}
real_periodic: indago.VariableDictType = {f'var{i}': (indago.VariableType.RealPeriodic, 0, 360) for i in range(0, dims)}
real_discrete: indago.VariableDictType = {f'var{i}': (indago.VariableType.RealDiscrete, [float(_) for _ in range(0, 361)]) for i in range (0, dims)}
real_discrete_periodic: indago.VariableDictType = {f'var{i}': (indago.VariableType.RealDiscretePeriodic, [float(_) for _ in range(0, 361)]) for i in range (0, dims)}
integer: indago.VariableDictType = {f'var{i}': (indago.VariableType.Integer, 0, 360) for i in range(0, dims)}
integer_periodic: indago.VariableDictType = {f'var{i}': (indago.VariableType.IntegerPeriodic, 0, 360) for i in range(0, dims)}

vars = real_periodic

def goalfun(x):
    x = np.asarray(x)
    # return np.sum((x - 0) ** 2)
    # return np.sum((x - 180) ** 2)
    offset = (1 + np.arange(np.size(x))) * 1.5
    return np.sum(np.sin(np.deg2rad(x - offset - 90))) + np.size(x)

def plotfun(ax):
    x, y = np.meshgrid(np.linspace(vars['var0'][1], vars['var0'][2], 101),
                       np.linspace(vars['var1'][1], vars['var1'][2], 101))
    pts = np.stack([x.ravel(), y.ravel()], axis=1)

    z = np.array([goalfun(p) for p in pts])
    z = z.reshape(x.shape)

    rx, ry = np.meshgrid(np.linspace(0, 1, 101),
                         np.linspace(0, 1, 101))
    ax.contourf(rx, ry, z, levels=50, cmap=plt.cm.jet, alpha=0.2)

optimizer = indago.PSO()
optimizer.variables = vars
optimizer.evaluator = goalfun  # minimum on bounds
optimizer._plot_evaluator = plotfun
optimizer.max_iterations = 100
optimizer.monitoring = 'basic'
# optimizer.optimize(seed=None)
# optimizer.plot_history()
# print(f'{optimizer.best.f=:.3e}')

import subprocess

# Frames per second
fps = 3
output_video = 'pso.mp4'
cmd = [
    "ffmpeg",
    "-y",  # overwrite output
    "-framerate", str(fps),
    "-i", 'it%03d.png',
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    output_video,
]
print(' '.join(cmd))
subprocess.run(cmd, cwd=f'/tmp/pso_periodic/', check=True)

print(f"Video saved to {output_video}")

plt.show()
