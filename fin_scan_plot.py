import defopt
import pandas as pd
import matplotlib.pylab as plt
import glob
import numpy as np


def main(*, input_dir: str='fin_scan'):
    """
    @input_dir input directory where results are stored
    """
    
    # gather all the results in a DataFrame
    alldfs = []
    for filename in glob.glob(input_dir + '/res*/results.csv'):
        df = pd.read_csv(filename)
        alldfs.append(df)
        
    if len(alldfs) == 0:
        print(f'Warning: not able to find any csv file!')
        return
    
    df = pd.concat(alldfs)
    df.to_csv(input_dir + '/scan.csv')
    
    print(df)
    print(df['normThickness'].unique())
    
    wstyles = [':', '-.', '-']
    widths = df['normThickness'].unique()
    widths.sort()
    rstyles = ['b', 'c', 'm', 'r']
    rvals = df['Re'].unique()
    rvals.sort()
    
    # add thrust
    df['L_over_D'] = df.lift/df.drag
    df['thrust'] = df.lift*np.sin(-df.alpha_rad) - df.drag*np.cos(-df.alpha_rad)

    for yname in 'lift', 'drag', 'thrust', 'L_over_D':
        plt.figure()
        legs = []
        for k in range(len(rvals)):
            for j in range(len(widths)):
                df2 = df[(df.Re == rvals[k]) & (df.normThickness == widths[j])]
                mk = rstyles[k % len(rvals)] + wstyles[j % len(widths)]
                x = -df2.alpha_rad*180/np.pi
                y = df2[yname]
                paired_list = sorted(zip(x, y))
                sorted_x, sorted_y = zip(*paired_list)
                plt.plot(sorted_x, sorted_y, mk)
                legs.append(f'Re={rvals[k]:.1e} thcknss={widths[j]:.3f}')
        plt.legend(legs)
        plt.xlabel('attack angle deg')
        plt.title(yname)
        plt.savefig(input_dir + f'/{yname}.png')
            
        

    
    
if __name__ == '__main__':
    defopt.run(main)
    