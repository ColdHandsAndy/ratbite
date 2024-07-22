![](docs/pics/banner.png)

Spectral Monte Carlo Path Tracer[^1] written in C++ with OptiX and CUDA  

### Feature list
* Hero Wavelength spectral sampling[^2]
* Sampling of visible normals[^3][^4]
* Conductor BxDF
* Dielectric BxDF (Reflection + Transmission)
* Depth of Field
* Low-discrepancy sequence generation
* Render preview

[^1]: [Physically Based Rendering: From Theory To Implementation. Fourth edition.](https://pbr-book.org/4ed/contents)
[^2]: [Hero Wavelength Spectral Sampling](https://cgg.mff.cuni.cz/~wilkie/Website/EGSR_14_files/WNDWH14HWSS.pdf)
[^3]: [Sampling Visible GGX Normals with Spherical Caps](https://arxiv.org/pdf/2306.05044)
[^4]: [Bounded VNDF Sampling for Smith–GGX Reflections](https://gpuopen.com/download/publications/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf)
