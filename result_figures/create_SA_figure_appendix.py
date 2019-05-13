import glob
import os
import numpy as np

fin = np.sort(glob.glob(os.path.join(os.path.sep, *['work', 'WVDP_EWG_STUDY3', 'study3py', 'result_figures', 'sensitivity_analysis', 'sew' , 'cat_of.morris_df_short.model_*.lowering.scatter.pdf'])))

fOut=r'sensitivity_analysis_appendix_figures.tex'
folder_name = 'SA_appendix_figures'
#%%
f = open(fOut, 'w')
#%%
sz='5'
for fname in fin:

    lowering_figure = os.path.split(fname)[-1].replace('.', '_', 4)

    lowering_filepath = folder_name + '/' + lowering_figure
    param_filepath = lowering_filepath.replace('lowering', 'parameter' )

    model_name = lowering_figure.split('_')[-3]
    print(model_name)


    f.write('% sensitivity analysis output for model number ' + model_name +'\n')
    f.write('\clearpage \n')
    f.write('\\begin{figure}% \n')
    f.write('    \centering \n')

    f.write(('    \subfloat[Input parameter sensitivity plot for model ' + model_name + ' in Upper Franks Creek Watershed (SEW domain). '
             'Colors represent  Method of Morris sensitivity analysis results for model input parameters. '
             'Shape represents postglacial topography and the two lowering histories considered are not distinguished. '
             'Thus two markers are present for each color-symbol combination.]{{\includegraphics[width=5.5in]{'+ param_filepath +'} }}% \n'))
    f.write('    \qquad \n')

    f.write("    \subfloat[Parameter, initial condition, and lowering sensitivity plot for model " + model_name + " in Upper Franks Creek Watershed (SEW domain). "
             "Colors represent parameter, initial condition, or lowering history sensitivities. The parameter sensitivities of the upper panel "
             "are shown in gray for context. Initial condition ``7\% etching'' and lowering history ``1'' were used as references value "
             "initial and boundary condition sensitivity calculations."
             "]{{\includegraphics[width=5.2in]{"+ lowering_filepath +"} }}% \n")

    f.write('    \qquad \n')


    f.write('        \caption{Sensitivity analysis summary for model ' + model_name + ' in Upper Franks Creek Watershed (SEW domain)}% \n')

    f.write('   \label{fig:sensitivity_appendix_' + model_name + '}% \n')
    f.write(' \end{figure}\n')
    f.write(' \n')
    f.write(' \n')





f.close()
