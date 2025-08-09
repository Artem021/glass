import os, re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
import multiprocessing as mp
import pwlf

KNOWN_TERMS = ['Bond', 'Angle', 'Fourier-Dih.', 'Ryckaert-Bell.', 'LJ-14', 'Coulomb-14', 'LJ-(SR)', 'Disper.-corr.', 'Coulomb-(SR)', 'Coul.-recip.', 'Potential', 'Kinetic-En.', 'Total-Energy', 'Conserved-En.', 'Temperature', 'Pres.-DC', 'Pressure', 'Box-X', 'Box-Y', 'Box-Z', 'Volume', 'Density', 'pV', 'Enthalpy', 'Vir-XX', 'Vir-XY', 'Vir-XZ', 'Vir-YX', 'Vir-YY', 'Vir-YZ', 'Vir-ZX', 'Vir-ZY', 'Vir-ZZ', 'Pres-XX', 'Pres-XY', 'Pres-XZ', 'Pres-YX', 'Pres-YY', 'Pres-YZ', 'Pres-ZX', 'Pres-ZY', 'Pres-ZZ', '#Surf*SurfTen', 'Box-Vel-XX', 'Box-Vel-YY', 'Box-Vel-ZZ', 'T-System']
UNITS = {'Bond': '(kJ/mol)', 'Angle': '(kJ/mol)', 'Ryckaert-Bell.': '(kJ/mol)', 'LJ-14': '(kJ/mol)', 'Coulomb-14': '(kJ/mol)', 'LJ (SR)': '(kJ/mol)', 'Disper. corr.': '(kJ/mol)', 'Coulomb (SR)': '(kJ/mol)', 'Coul. recip.': '(kJ/mol)', 'Potential': '(kJ/mol)', 'Kinetic En.': '(kJ/mol)', 'Total Energy': '(kJ/mol)', 'Conserved En.': '(kJ/mol)', 'Temperature': '(K)', 'Pres. DC': '(bar)', 'Pressure': '(bar)', 'Constr. rmsd': '()', 'Box-X': '(nm)', 'Box-Y': '(nm)', 'Box-Z': '(nm)', 'Volume': '(nm^3)', 'Density': '(kg/m^3)', 'pV': '(kJ/mol)', 'Enthalpy': '(kJ/mol)', 'Vir-XX': '(kJ/mol)', 'Vir-XY': '(kJ/mol)', 'Vir-XZ': '(kJ/mol)', 'Vir-YX': '(kJ/mol)', 'Vir-YY': '(kJ/mol)', 'Vir-YZ': '(kJ/mol)', 'Vir-ZX': '(kJ/mol)', 'Vir-ZY': '(kJ/mol)', 'Vir-ZZ': '(kJ/mol)', 'Pres-XX': '(bar)', 'Pres-XY': '(bar)', 'Pres-XZ': '(bar)', 'Pres-YX': '(bar)', 'Pres-YY': '(bar)', 'Pres-YZ': '(bar)', 'Pres-ZX': '(bar)', 'Pres-ZY': '(bar)', 'Pres-ZZ': '(bar)', '#Surf*SurfTen': '(bar nm)', 'Box-Vel-XX': '(nm/ps)', 'Box-Vel-YY': '(nm/ps)', 'Box-Vel-ZZ': '(nm/ps)', 'T-System': '(K)'}

def reduce_data(xdata, ydata, xlo, xhi):
    if xlo > xhi:
        print(f'ERROR (reduce_data) no data between {xlo}-{xhi}')
    reducedx = []; reducedy = [];  
    for x, y in zip(xdata, ydata):
        if x >= xlo and x <= xhi:
            reducedx.append(x); reducedy.append(y);
    return reducedx, reducedy

def _fit_hyperbola(x, y, xlo, xhi, minimum_convergence=None, initial_guess=False, maxiter=10**4):
    '''
    Adapted from Lunar code: https://github.com/CMMRLab/LUNAR
    
    Fits 2D data to hyperbolic curve according to method proposed in paper: https://doi.org/10.1016/j.polymer.2016.01.074
    '''
      
    # define default outputs (in-case something goes wrong)
    xout = list(x); yout = len(y)*[0];
    params = [0, 0, 0, 0, 0]; center = [0, 0];
    slopes = [0, 0]; transition = [0, 0];
    
    # Convert to float64 and then lists to numpy arrays
    xx = np.array(x); yy = np.array(y);
    
    # Setup intial guess
    p0 = None
    if initial_guess:
        slopeguess = (yy[-1]-yy[0])/(xx[-1]-xx[0])
        p0 = (np.mean(xx),np.mean(yy), slopeguess, slopeguess, np.log((xx[-1]-xx[0])**2/100))

    # Define the hyperbola equation (eqn 1 in paper)
    def hyberbola(t, t0, v0, a, b, c):
        h0 = 0.5*(t-t0) + np.sqrt( ((t-t0)*(t-t0))/4 + np.exp(c) )
        v = v0 + a*(t-t0) + b*h0
        return v
    
    # Find best fit    t0          v0        a        b        c
    parm_bounds = ((np.min(xx), np.min(yy), -np.inf, -np.inf, -np.inf), # lower
                   (np.max(xx), np.max(yy),  np.inf,  np.inf,  np.inf)) # upper
    param, param_cov = optimize.curve_fit(hyberbola, xx, yy, p0=p0, method='trf', bounds=parm_bounds, maxfev=maxiter)
    t0, v0, a, b, c = param # extract out fitting coeffs
    
    # update defaults
    params = list(param)
    yout = list(hyberbola(xout, *param))
    center = [t0, v0]
    slopes = [a, (a+b)]
    
    # Find where function is transitioning (eqn 5 in paper)
    if minimum_convergence is not None:
        dtemp = (np.exp(c/2)*(2*minimum_convergence-1))/(np.sqrt(minimum_convergence*(1-minimum_convergence)))
        transition = [t0-dtemp, t0+dtemp]
    return xout, yout, params, center, slopes, transition

def fit_hyperbola(x, y, xlo, xhi, minimum_convergence=None, initial_guess=False, maxiter=10**4):
    # xlo = min(x); xhi = max(x) # use all available data
    reduced_x, reduced_y = reduce_data(x, y, xlo, xhi)
    if reduced_x and reduced_y:        
        xout, yout, params, center, slopes, transition = _fit_hyperbola(x, y, xlo, xhi, minimum_convergence, initial_guess, maxiter)
    else:
        xout = [0, 1]; yout = [0, 1]; slopes = [0, 1]
        center = [0, 0]; params = [0, 0, 0, 0, 0];
        transition = [];
        raise ValueError(f'ERROR no (hyperbola) LAMMPS data in xrange {xlo} - {xhi}')
    return xout, yout, params, center, slopes, transition

def _piecewise_regression(x, y, xlo, xhi, n):
    
    npx = np.array(x); npy = np.array(y);
    
    # Perform peicewise regression
    my_pwlf = pwlf.PiecewiseLinFit(npx, npy)
    xbreaks = my_pwlf.fit(n+1)
    ybreaks = my_pwlf.predict(np.array(xbreaks))
    xout = np.linspace(npx.min(), npx.max(), 100)
    yout = my_pwlf.predict(xout)
    
    # Compute the slopes
    def slope(x1, y1, x2, y2):
        m = (y2-y1)/(x2-x1)
        return m
    
    npoints = min([len(xbreaks), len(ybreaks)])-1
    slopes = {(i, i+1):slope(xbreaks[i], ybreaks[i], xbreaks[i+1], ybreaks[i+1]) for i in range(npoints)}
    return xout, yout, xbreaks, ybreaks, slopes

def piecewise_regression(x, y, xlo, xhi, n):
    # xlo = min(x); xhi = max(x) # use all available data
    reduced_x, reduced_y = reduce_data(x, y, xlo, xhi)
    if reduced_x and reduced_y:
        xout, yout, xbreaks, ybreaks, slopes = _piecewise_regression(reduced_x, reduced_y, xlo, xhi, n)
    else:
        xout = [0, 1]; yout = [0, 1]; slopes = {(0,1):1}
        xbreaks = [0, 1]; ybreaks = [0, 1]
        raise ValueError(f'ERROR no (peicewise-regression) LAMMPS data in xrange {xlo} - {xhi}')
    return xout, yout, xbreaks, ybreaks, slopes

# https://stackoverflow.com/questions/43925337/matplotlib-returning-a-plot-object
def xvg2D(xvgfile : str, title = '', xlab = '', ylab = '', style = 'line',  add_mean = False):
    _avail_styles = ['line', 'scatter']
    assert style in _avail_styles, f'Available styles are : {_avail_styles}'
    _tp = re.compile(r'@\s+title\s+"(.+)"')
    _xp = re.compile(r'@\s+xaxis\s+label\s+"(.+)"')
    _yp = re.compile(r'@\s+yaxis\s+label\s+"(.+)"')
    _vp = re.compile(r'\s+legend\s+"(.+)"')
    assert os.path.exists(xvgfile), 'file not found'
    with open(xvgfile,'r') as fi:
        cont = fi.read()
    x, y = np.loadtxt(xvgfile, comments=["@", "#", "&"], unpack=True)
    if style == 'line':
        plt.plot(x, y)
    elif style == 'scatter':
        plt.scatter(x, y, s=0.1)
    if add_mean:
        plt.axhline(y=y.mean(), color='r', linestyle='--', label=f'average = {y.mean():.2f}')
        # plt.annotate(f'{y.mean():.2f}', (x.mean(), y.mean()+10), fontsize=10, color='b')
        plt.legend()
        # print('plot with mean, label = ', f'{y.mean():.2f}')
    if xlab == '':
        try:
            xlab = re.search(_xp, cont).group(1)
        except:
            print('Warning: could not parse X labels from xvg')
    if ylab == '':
        try:
            yunit = re.search(_yp, cont).group(1)
            yval = re.search(_vp, cont).group(1)
            ylab = ' '.join((yval, yunit))
        except:
            print('Warning: could not parse Y labels from xvg')
    if title == '':
        try:
            title = re.search(_tp, cont).group(1)
        except:
            print('Warning: could not parse title labels from xvg')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.show()
    # plt.savefig('xvgplot.png')

def pd2D(df : pd.DataFrame, xval : str = 'Temperature', yval : str = 'Volume', hyperbolic_fit = False, bilinear_fit = False, linear_fit = False, n = 5, title = None, s : float = 0.1, save = True, show = False):
    '''
    2D plot of columns `xval` and `yval` in dataframe `df`. Strings `xval` and `yval` may be substrings of real column names, 
    for example, 'Temperature' will match 'Temperature (K)' column. If guess fails, AssertionError will be raised.
    '''
    col_names = df.columns.to_list()
    xlab = next((s for s in col_names if xval in s), None)
    ylab = next((s for s in col_names if yval in s), None)
    assert xlab != None and ylab !=None, f'Properties "{xval}", "{yval}" not found in dataframe: {col_names}'
    x = df[xlab].to_numpy()
    y = df[ylab].to_numpy()
    xstart = x[: x.size//1000].mean()
    xend = x[-x.size//1000 :].mean()
    if xstart > xend:
        xstart = x[-x.size//1000 :].mean()
        xend = x[: x.size//1000].mean()
    plt.scatter(x, y, s=s, marker='o', color='gray', label = 'Raw data')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if hyperbolic_fit:
        try:
            # xout, yout, params, center, slopes, transition
            xh, yh, _parms, center, sl, _tr = fit_hyperbola(x, y, xstart, xend)
            # plt.axvline(x=center[0], color='r', linestyle='--', label=f'xfit = {center[0]:.2f}')
            plt.plot(xh, yh, color = 'blue', label = 'Hyperbolic fit')
            plt.plot([min(xh), center[0], max(xh)], [min(yh), center[1], max(yh)], color = 'blue', linestyle='--')
            plt.plot(*center, 'bo', label = f'Center of h-fit: ({center[0]:.1f}, {center[1]:.1f})')
        except:
            print('Warning: hyperbolic fit failed and will not be added to plot')
    if bilinear_fit:
        try:
            xh, yh, xbreaks, ybreaks, sl = piecewise_regression(x, y, xstart, xend, n=1)
            plt.plot(xh, yh, color = 'red', label = 'Bilinear fit')
            plt.plot(xbreaks[1], ybreaks[1], 'ro', label = f'Center of bl-fit: ({xbreaks[1]:.1f}, {ybreaks[1]:.1f})')
        except:
            print('Warning: bilinear fit failed and will not be added to plot')
    if linear_fit:
        # try:
        xstart = x[: x.size//1000].mean()
        xend = x[-x.size//1000 :].mean()
        x0 = x[: x.size//n] # first 1/n
        x1 = x[-x.size//n :] # last (n-1)/n
        y0 = y[: y.size//n] # first 1/n
        y1 = y[-y.size//n :] # last (n-1)/n
        k0, b0 = np.polyfit(x0,y0, 1)
        k1, b1 = np.polyfit(x1,y1, 1)
        ix = -(b0 - b1) / (k0 - k1) # x intersection point
        iy = ix * k0 + b0 # y intersection point
        # print(ix,iy)
        xfit = np.array([xstart, ix, xend]) # fitted x points
        yfit = np.array([xstart*k0+b0, iy, xend*k1+b1]) # fitted y points
        plt.plot(xfit, yfit, color = 'green', label = f'Linear fit, n = {n}')
        plt.plot(ix,iy, 'go', label = f'Center of l-fit: ({ix:.1f}, {iy:.1f})')
        # except:
        #     print('Warning: linear fit failed and will not be added to plot')
    
    
    plt.legend()
    if title != None:
        plt.title(title)
    if save:
        plt.savefig(f'{yval}_{xval}.png')
    if show:
        plt.show()


# TODO
def compute_transition(x : np.ndarray, y : np.ndarray, minx : float, maxx : float, method = 'linear4', n = 4):
    '''
    Find transition point on 2D dataset.
    
    '''
    _avail = ['hyperbola', 'piecewise', 'linear4']
    assert(method in _avail), f'Unknown method requested: {method}. Available methods are: ' + ', '.join(_avail)

    if method == 'linear4':
        sel = np.isfinite(x)
        if minx != None:
            sel = sel & (x>=minx)
        if maxx != None:
            sel = sel & (x<=maxx)
        # sel = (x>=minx) & (x <= maxx)
        x = x[sel]
        y = y[sel]
        plt.scatter(x, y, s=0.1)
        x0 = x[: x.size//n] # first 1/n
        x1 = x[-x.size//n :] # last (n-1)/n
        y0 = y[: y.size//n] # first 1/n
        y1 = y[-y.size//n :] # last (n-1)/n
        k0, b0 = np.polyfit(x0, y0, 1)
        k1, b1 = np.polyfit(x1, y1, 1)
        ix = -(b0 - b1) / (k0 - k1) # x intersection point
        iy = ix * k0 + b0 # y intersection point
        ylfit = np.polyval((k0, b0), x0)
        yrfit = np.polyval((k1, b1), x1)
        plt.plot(x0, ylfit, color = 'black', label = f'Linear fit, n = {n}')
        plt.plot(x1, yrfit, color = 'black')
        plt.plot(ix,iy, 'go', label = f'Center of l-fit: ({ix:.1f}, {iy:.1f})')
        plt.legend()
        plt.show()
        return (ix,iy)
    elif method == 'piecewise':
        pass
    

def xvg_to_pd(xvgfile : str):
    '''
    Convert xvg file generated by `gmx energy` to Pandas dataframe.
    '''
    assert os.path.exists(xvgfile), 'file not found'
    _tp = re.compile(r'@\s+title\s+"(.+)"')
    _xp = re.compile(r'@\s+xaxis\s+label\s+"(.+)"')
    _yp = re.compile(r'@\s+yaxis\s+label\s+"(.+)"')
    _vp = re.compile(r'\s+legend\s+"(.+)"')
    with open(xvgfile,'r') as fi:
        cont = fi.read()
    xlab = re.search(_xp, cont).group(1)
    yunit = re.search(_yp, cont).group(1)
    yval = re.findall(_vp, cont)
    if yunit == 'Free Volume (%)':
        x, fv, v = np.loadtxt(xvgfile, comments=["@", "#", "&"], unpack=True)
        return pd.DataFrame({xlab: x, yunit: fv, 'Volume (nm^3)' : v})
    nycols = len(yval)
    if nycols == 0:
        raise ValueError(f'`legend "XXX"` record was expected but not found in xvg file: {xvgfile}')
    elif nycols == 1:
        cnames = [xlab, ' '.join([yval[0], yunit])]
    else:
        cnames = [xlab] + [' '.join([s, UNITS[s]]) for s in yval]
    vals = np.loadtxt(xvgfile, comments=["@", "#", "&"], unpack=True)
    assert len(cnames) == len(vals), 'number of labels does not match number of columns'
    return pd.DataFrame(dict(zip(cnames, vals)))
    
def merge_xvg(filelist : list[str]):
    '''
    Merges multiple xvg files by 'Time' column and returns single dataframe.
    
    '''
    from functools import reduce
    dfs_to_merge = []
    xvglist = [f for f in filelist if f.endswith('xvg')]
    if len(filelist) != len(xvglist):
        print(f'Warning: files that do not have .xvg extension will be ignored: {set(filelist) - set(xvglist)}')
    for xvgfile in xvglist:
        try:
            df = xvg_to_pd(xvgfile)
            dfs_to_merge.append(df)
        except AssertionError:
            print(f'Skip file {xvgfile}: file not exists')
    if len(dfs_to_merge) == 0:
        raise ValueError('Nothing to merge')
    time_col = next((s for s in dfs_to_merge[0].columns.tolist() if 'Time' in s), None)
    if time_col == None:
        raise ValueError('Dataframe does not contain column with time coordinate')
    print(f'merging {len(dfs_to_merge)} dataframes by "{time_col}" column')
    return reduce(lambda x,y: pd.merge(x, y, on = time_col, how = 'outer'), dfs_to_merge)


def select_columns(df : pd.DataFrame, properties : list[str]):
    '''
    Select columns in dataframe with `gmx energy` that match parameters (unitless). For example, 
    `Time` match column `Time (ps)` in dataframe. Returns subset of provided dataframe, containing only 
    requested columns. Raises AssertionError if unknown property requested. Available properties are listed below:
    
    'Time', 'Density', 'Volume', 'Potential', 'Temperature'
    '''
    _avail = ['Time', 'Density', 'Volume', 'Potential', 'Temperature']
    _full_cnames = []
    cnames = df.columns.tolist()
    if isinstance(properties, str):
        properties = [properties]
    properties = [p.strip() for p in properties]
    assert set(properties).issubset(set(_avail)), f'Unknown properties requested: {set(properties) - set(_avail)}'
    for p in properties:
        full_pname = next((s for s in cnames if p in s), None)
        if full_pname == None:
            print(f'Warning: property "{p}" not present in dataframe')
            continue
        _full_cnames.append(full_pname)
    return df[_full_cnames]


TEST = False
if __name__ == '__main__':
    cool = r'C:\Users\artem\OneDrive\_Work_Kunitsyn\Projects\PolyBMSTU\Tg_calculations\pvc\7k\rate20\cool\1.lammps.csv'
    df = pd.read_csv(cool)
    x = df['Temp'].to_numpy()
    y = df['Volume'].to_numpy()
    # plt.scatter(x, y, s=0.2)
    # plt.xlabel('Temperature (K)')
    # plt.ylabel('Volume, A^3')
    print(compute_transition(x,y,210,490))
    # plt.show()
    print(0)
    
    
    if TEST:
        print('nothing to test yet')
        # print(free_volume(wd))
        
