
import os
import sys
import pickle
import pandas as pd
import numpy as np


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] in ('-h', '--help'):
        print(f'Usage: {os.path.basename(sys.argv[0])} pkl_file')
        sys.exit(1)
    
    pkl_file = sys.argv[1]
    data = pickle.load(open(pkl_file, 'rb'))
    coords = data['centers']
    impedance = data['R']
    time = data['time']
    Vm = data['Vm']
    indexes = data['segment_indexes']
    indexes['soma'] = [0]

    xls_file = os.path.splitext(pkl_file)[0] + '.xlsx'

    with pd.ExcelWriter(xls_file) as writer:
        for loc in coords:
            for i, (point, R, index, t, v) in enumerate(zip(coords[loc], impedance[loc], indexes[loc], time[loc], Vm[loc])):
                df1 = pd.DataFrame({'x': point[0], 'y': point[1], 'z': point[2], 'impedance': R}, index=[0])
                df2 = pd.DataFrame({'t': t, 'Vm': v})
                df = pd.concat([df1, df2], ignore_index=True, axis=1)
                df.columns = 'x', 'y', 'z', 'impedance', 'Time', 'Vm'
                df.to_excel(writer, sheet_name=f'{loc}_{index}')
                sys.stdout.write('.')
                sys.stdout.flush()
                if (i+1) % 50 == 0:
                    sys.stdout.write('\n')
            if (i+1) % 50 != 0:
                sys.stdout.write('\n')

