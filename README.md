# ARFF-KLMS Experiment

This project simulates an experiment using Adaptive Random Fourier Features Kernel Least Mean Squares (ARFF-KLMS) to approximate non-linear functions. The ARFF-KLMS algorithm adapts the kernel bandwidth online, enhancing tracking and convergence in non-stationary environments. The experiment calculates and plots the mean squared error (MSE) over iterations to evaluate the model's performance.

This simulation is inspired by the work of Wei Gao, Jie Chen, Cédric Richard, Wentao Shi, and Qunfei Zhang, titled "Adaptive Random Fourier Features Kernel LMS." [Read the paper](
https://doi.org/10.48550/arXiv.2207.07236).

## Project Structure

The project is divided into five main scripts:

1. **data_generator.py**
   - Contains the `generate_synthetic_data` function, which generates synthetic data for the experiment.

2. **random_feature_model.py**
   - Contains the `RandomFeatureModel` class, which represents the ARFF-KLMS model and performs training.

3. **experiment.py**
   - Contains the `run_experiment` function, which runs the experiment and collects MSE values.

3. **plotter.py**
   - Contains the `plot_mse` function, which plots the MSE values.

3. **main.py**
   - The main script that orchestrates the experiment, collects MSE values, and plots the results.

## Installation

1. Clone the repository:
    
```
    git clone https://github.com/Parth-nXp/ARFF-KLMS.git
    cd ARFF-KLMS
```

2. Create a virtual environment and activate it:
    
```
    python -m venv rff-klms-env
    source rff-klms-env/bin/activate  # On Windows use rff-klms-env\Scripts\activate
```

3. Install the required packages:
    
```
    pip install -r requirements.txt
```

## Usage

Run the main script to start the experiment:
```
python main.py
```

## Troubleshooting

If you encounter any issues or errors while running the project, please check the following:

- Ensure all dependencies are installed correctly by running pip install -r `requirements.txt`.
  
- Make sure you are using a compatible version of Python (e.g., Python 3.6 or higher).
 
- If you encounter issues related to missing files or incorrect paths, verify that you are in the correct directory (`ARFF-KLMS`).

If problems persist, feel free to open an issue on GitHub.

## Contributing

Contributions are welcome! If you have suggestions for improvements or bug fixes, please follow these steps:

1. Fork the repository.

2. Create a new branch (`git checkout -b feature-branch`).

3. Make your changes and commit them (`git commit -m 'Add some feature'`).

4. Push to the branch (`git push origin feature-branch`).

5. Open a pull request.

Please ensure your code follows the existing style and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details.
