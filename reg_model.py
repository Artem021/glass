import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

def rmse(a : np.ndarray, b : np.ndarray):
    return np.sqrt(((a - b)**2).mean())

def r2(x,y):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2

def plot_df(df, xlab='Temp', ylab='Volume', xstart=200, xend=500, n=4):
    x = df[xlab].to_numpy()
    y = df[ylab].to_numpy()
    dx = abs(xstart-xend)

    x_low = x[(x >= xstart) & (x <= xstart + dx/n)]
    y_low = y[(x >= xstart) & (x <= xstart + dx/n)]
    x_high = x[(x >= xend - dx/4) & (x <= xend)]
    y_high = y[(x >= xend - dx/4) & (x <= xend)]

    ml, bl = np.polyfit(x_low, y_low, 1)
    mh, bh = np.polyfit(x_high, y_high, 1)

    plt.scatter(x, y, s=0.1, label = 'raw data')
    plt.plot(x_low, ml*x_low+bl, '--k')
    plt.plot(x_high, mh*x_high+bh, '--k')
    plt.show()

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def moving_average2(x, w):
  return np.convolve(x, np.ones(w), 'valid') / w

# f = r'C:\Users\artem\OneDrive\_Work_Kunitsyn\Projects\PolyBMSTU\Tg_calculations\pvc\7k\rate20\cool\2.lammps.csv'
# df = pd.read_csv(f, sep=',')
# plot_df(df)

# f = r'C:\Users\artem\OneDrive\_Work_Kunitsyn\Projects\PolyBMSTU\Tg_calculations\pvc\7k\rate20\cool\3.lammps.csv'
# df = pd.read_csv(f, sep=',')
# plot_df(df)

# f = r'C:\Users\artem\OneDrive\_Work_Kunitsyn\Projects\PolyBMSTU\Tg_calculations\pvc\7k\rate20\cool\4.lammps.csv'
# df = pd.read_csv(f, sep=',')
# plot_df(df)

# f = r'C:\Users\artem\OneDrive\_Work_Kunitsyn\Projects\PolyBMSTU\Tg_calculations\pvc\7k\rate20\cool\5.lammps.csv'
# df = pd.read_csv(f, sep=',')
# plot_df(df)
f = r'C:\Users\artem\OneDrive\_Work_Kunitsyn\Projects\PolyBMSTU\Tg_calculations\pvc\7k\rate20\cool\1.lammps.csv'
df = pd.read_csv(f, sep=',')
x = df['Temp'].to_numpy()
y = df['Volume'].to_numpy()

# n = 1000
# plt.plot(x,y)
# ys = scipy.signal.savgol_filter(y, n, 3)
# plt.plot(x, ys, color = 'red', linestyle='solid')
# plt.plot(x[n:], moving_average(ys,n+1), color='black')
# plt.show()

# plot_df(df)
# plt.scatter(x,y, s=0.1)
# plt.xlabel('Temperature, K')
# plt.ylabel('Volume, A^3')
# plt.show()
# exit()

tstart = 200
tend = 500
dt = abs(tend-tstart)

x_err = []
x_window = []
xchunk = 2 # K
for i in range(dt//xchunk):
    win = tstart + xchunk*(i+1)
    cx = x[x <= win]
    cy = y[x <= win]
    b, m = np.polyfit(cx, cy, 1)
    pv = np.polyval(np.polyfit(cx, cy, 1), cx)
    # print(pv.size, cx.size)
    # plt.plot(cx, pv)
    # plt.show()
    mse = ((pv - cy)**2).mean()
    _rmse = np.sqrt(((pv - cy)**2).mean())
    # plt.plot(cx, np.polyval(np.polyfit(cx, cy, 1), cx), label=f'({tstart}, {win}) K; err = {se:.1f}')
    x_window.append(win)
    x_err.append(_rmse)
    # x_err.append(mse)
    # x_err.append(r2(cx,cy))
x_err = np.array(x_err)
min_e = x_err.min()
npoint = int(abs(tend - tstart)) * 100
print(npoint)
xi = np.linspace(tstart, tend, npoint)
yi = np.interp(xi, x_window, x_err)
# plt.plot(xi, yi)
# ys = scipy.signal.savgol_filter(yi, 100, 3)
# plt.plot(xi, scipy.signal.savgol_filter(yi, 33, 3), color = 'black', linestyle='dashed')
# plt.plot(xi, np.gradient(yi,1), color = 'red', linestyle='dashed')
# plt.plot(xi, np.gradient(yi,2), color = 'orange', linestyle='dashed')
# plt.plot(xi, np.gradient(yi,5), color = 'yellow', linestyle='dashed')
# plt.plot(xi, np.gradient(yi,10), color = 'green', linestyle='dashed')
plt.plot(xi, np.gradient(yi, 10), color = 'pink', linestyle='dashed')


# plt.plot(xi, yi)
plt.xlabel('Temperature window, K')
plt.ylabel("RMSE' of linear fit")
plt.show()
exit()
print(0)

# peaks, _ = scipy.signal.find_peaks(np.array(x_err)*-1)
# peaks, _ = scipy.signal.find_peaks(yi)
# plt.plot(xi,yi)
# plt.show()
# print(peaks)
# peaks, _ = scipy.signal.find_peaks(yi*-1)
# print(peaks)
# peaks = scipy.signal.argrelmin(np.array(x_err))
# print(x_err[peaks])
# plt.plot(x_window[peaks], x_err[peaks], "x")
plt.plot(x_window[x_err.index(min_e)], min_e, label = f'min error = {min_e:.1f}', marker = 'o', color = 'red')
plt.plot(x_window, x_err)
plt.scatter(x_window, x_err)
plt.xlabel('t_max')
plt.ylabel('r2')


# plt.scatter(x, y, s=0.1)
# plt.plot(x_low, ml*x_low+bl, '--k')
# plt.plot(x_high, mh*x_high+bh, '--k')
plt.legend()
plt.show()

