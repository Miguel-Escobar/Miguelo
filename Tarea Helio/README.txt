Summary of fortran codes aimed to obtain bound states 
identifying singularities of the t matrix.
Prepared by HFArellano for 1lecture in FI4005, 9nov2022

bstates01_arellano.f
To calculate and print square well potential in momentum space.
Output in fort.7, ready to plot with gnuplot (splot)


bstates02_arellano.f
To calculate det(1 - V*G0(z)) as function of the energy z.
Output in fort.8: energy, det, 1/det

bstates03_arellano.f
To calculate eigenenergies imposing det(1 - V*G0(z))
Illustration of halving technique

bstates04_arellano.f
To calculate eigenenergies imposing det(1 - V*G0(z))

bstates05_arellano.f
To calculate eigenfunctions at specified n-level
 fort.10 contains k vs |<k|v|Psi>|^2
 fort.12 contains k', k, vs <k'|v|Psi><Psi|v|k>   (surface)
 fort.14 contains k', k, vs <k'|Psi><Psi|k>       (surface)
 fort.16 contains  k, vs |<k|Psi>|^2

