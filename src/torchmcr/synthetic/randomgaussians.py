import numpy as np
def generate_spectra(M_spectra,
                     N_peaks,
                     N_waves=1000,
                     x_range=(0, 1000),
                     center_limits=(100, 900),
                     variance_limits=(1, 50),
                     amplitude_limits=(0.5, 1.5)):
    """
    Generate M spectra with N Gaussian peaks.

    Parameters:
        M (int): Number of spectra to generate.
        N (int): Number of Gaussian peaks per spectrum.
        x_range (tuple): The range of x values for the spectra.
        center_limits (tuple): Limits for the random centers of Gaussian peaks.
        variance_limits (tuple): Limits for the random variances of Gaussian peaks.
        amplitude_limits (tuple): Limits for the random amplitudes of Gaussian peaks.

    Returns:
        np.ndarray: Array of shape (M, len(x)) containing the generated spectra.
    """
    # Generate x values within the specified range
    x = np.linspace(x_range[0], x_range[1], N_waves)
    spectra = np.zeros((M_spectra, N_waves))

    # Generate each spectrum
    for i in range(M_spectra):
        for _ in range(N_peaks):
            center = np.random.uniform(*center_limits)
            variance = np.random.uniform(*variance_limits)
            amplitude = np.random.uniform(*amplitude_limits)
            gaussian = amplitude * np.exp(-((x - center) ** 2) / (2 * variance ** 2))
            spectra[i] += gaussian

    return x, spectra
