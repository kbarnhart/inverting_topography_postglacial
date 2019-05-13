import glob
import os
import numpy as np


fin = np.sort(glob.glob(os.path.join(os.path.sep, *['work', 'WVDP_EWG_STUDY3', 'study3py', 'result_figures', 'calibration', 'sew' , 'BEST_PARAMETERS', '*.diff_modern.png'])))

fOut=r'calibration_appendix_figures.tex'
folder_name = 'calibration_appendix_figures'
#%%
f = open(fOut, 'w')
#%%
sz='5'
for fname in fin:

    diff_modern = os.path.split(fname)[-1].replace('.', '_', 4)

    diff_modern_filepath = folder_name + '/' + diff_modern
    
    topo_filepath = diff_modern_filepath.replace('_diff_modern', '' )
    eff_resid_filepath = diff_modern_filepath.replace('diff_modern', 'eff_resid' )
    change_filepath = diff_modern_filepath.replace('diff_modern', 'elevation_change_since_start')

    model_name = diff_modern_filepath.split('_')[-8]
    
    print(model_name)


    f.write('% calibration output for model number ' + model_name +'\n')
    f.write('\clearpage \n')
    f.write('\\begin{figure}% \n')
    f.write('    \centering \n')

    f.write(('    \subfloat[Modeled modern topography.'
             '] {{\includegraphics[width=3in]{'+ topo_filepath +'} }}% \n'))

    f.write(('    \subfloat[Modeled modern topography minus actual modern topography. Purple indicates that modeled topography is above actual topography and orange indicates that modeled topography is below actual topography.'
             '] {{\includegraphics[width=3in]{'+ diff_modern_filepath +'} }} \\ % \n'))
    
    f.write(('    \subfloat[Cumulative erosion from 13 ka to modern. Red indicates that erosion occurred, and blue indicates that deposition occurred.'
             '] {{\includegraphics[width=3in]{'+ change_filepath +'} }}% \n'))
    
    f.write(('    \subfloat[Effective residual value at each grid node used in for objective function calculation.'
             '] {{\includegraphics[width=3in]{'+ eff_resid_filepath +'} }}% \n'))


    f.write(('        \caption{Calibration results summary for Model ' + model_name + ' showing spatially distributed values at the end of the 13 ka to present model run with calibrated parameter values in Upper Franks Creek Watershed (SEW domain).}% \n'))

    f.write('   \label{fig:calib_appendix_' + model_name + '}% \n')
    f.write(' \end{figure}\n')
    f.write(' \n')
    f.write(' \n')





f.close()
