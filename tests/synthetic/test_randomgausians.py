import pytest
import numpy as np
from torchmcr.synthetic.randomgaussians import generate_spectra  # replace 'your_module' with the name of the module where generate_spectra is located

def test_generate_spectra():
    # Define parameters for the test
    M_spectra = 5
    N_peaks = 10
    N_waves = 1000
    x_range = (0, 1000)
    center_limits = (100, 900)
    variance_limits = (1, 50)
    amplitude_limits = (0.5, 1.5)

    # Call the function
    x, spectra = generate_spectra(M_spectra, N_peaks, N_waves, x_range, center_limits, variance_limits, amplitude_limits)

    # Check that the x array has the correct length
    assert len(x) == N_waves, f"Expected x to have length {N_waves}, but got {len(x)}"

    # Check that the spectra array has the correct shape
    assert spectra.shape == (M_spectra, N_waves), f"Expected spectra to have shape {(M_spectra, N_waves)}, but got {spectra.shape}"

    # Check that x values are within the specified range
    assert x[0] == pytest.approx(x_range[0]), "x values do not start within the specified x_range"
    assert x[-1] == pytest.approx(x_range[1]), "x values do not end within the specified x_range"

    # Check that the spectra values are non-negative
    assert np.all(spectra >= 0), "Spectra values should be non-negative"

    # Check that each spectrum has values close to zero outside peak areas
    # Generate an example Gaussian to estimate the spread
    example_gaussian = np.exp(-((x - 500) ** 2) / (2 * 50 ** 2))  # centered at 500, variance ~50
    threshold = 0.01 * np.max(example_gaussian)  # threshold below which values should be ~zero
    outer_band = np.logical_or(x < center_limits[0] - 3 * variance_limits[1],
                               x > center_limits[1] + 3 * variance_limits[1])
    assert np.all(spectra[:, outer_band] < threshold), "Expected outer bands to be close to zero"

# Run the test with pytest
if __name__ == "__main__":
    pytest.main()
