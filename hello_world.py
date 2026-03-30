import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def generate_data(n_points=50, x_min=-2, x_max=2, noise_std=0.5, seed=42):
    """
    Generate synthetic data: y = x^3 + noise
    
    Args:
        n_points: Number of data points
        x_min: Minimum x value
        x_max: Maximum x value
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed
        
    Returns:
        Tuple of (x, y) arrays
    """
    print("Generating synthetic data...")
    np.random.seed(seed)
    x = np.linspace(x_min, x_max, n_points)
    noise = np.random.normal(0, noise_std, len(x))
    y = x**3 + noise
    return x, y


def save_data(df, filepath='data/data.csv'):
    """
    Save dataframe to CSV file.
    
    Args:
        df: Pandas dataframe
        filepath: Path to save the CSV file
    """
    print(f"Creating and saving dataframe...")
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")


def load_data(filepath='data/data.csv'):
    """
    Load dataframe from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Pandas dataframe
    """
    print("Loading dataframe...")
    df = pd.read_csv(filepath)
    print(f"Loaded data shape: {df.shape}")
    print(df.head())
    return df


def fit_data(df, degree=2):
    """
    Fit polynomial regression model to data.
    
    Args:
        df: Pandas dataframe with 'x' and 'y' columns
        degree: Degree of polynomial
        
    Returns:
        Tuple of (poly, model, r_squared, X, y_data)
    """
    print(f"\nFitting polynomial model (degree {degree})...")
    
    # Prepare data
    X = df['x'].values.reshape(-1, 1)
    y_data = df['y'].values
    
    # Fit model
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y_data)
    
    # Calculate R-squared score
    r_squared = model.score(X_poly, y_data)
    
    # Report results
    print("\n" + "="*50)
    print("MODEL RESULTS")
    print("="*50)
    print(f"Fitted equation: y = {model.intercept_:.4f} + {model.coef_[1]:.4f}*x + {model.coef_[2]:.4f}*x²")
    print(f"R-squared score: {r_squared:.6f}")
    print("="*50)
    
    return poly, model, r_squared, X, y_data


def plot_results(df, poly, model, r_squared, output_file='plot.png'):
    """
    Create and save visualization of data and fitted model.
    
    Args:
        df: Pandas dataframe with 'x' and 'y' columns
        poly: PolynomialFeatures transformer
        model: Fitted LinearRegression model
        r_squared: R-squared score
        output_file: Path to save the plot
    """
    print("\nCreating visualization...")
    plt.figure(figsize=(10, 6))
    
    # Plot original data
    plt.scatter(df['x'], df['y'], alpha=0.6, s=50, 
                label='Data (y = x² + noise)', color='blue')
    
    # Generate and plot fitted curve
    x_fit = np.linspace(-2, 2, 200).reshape(-1, 1)
    X_fit = poly.transform(x_fit)
    y_fit = model.predict(X_fit)
    plt.plot(x_fit, y_fit, 'r-', label='Fitted model', linewidth=2)
    
    # Plot the true function
    y_true = x_fit.flatten()**3
    plt.plot(x_fit, y_true, 'g--', label='True function (y = x²)', linewidth=2, alpha=0.7)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(f'Polynomial Regression Fit (R² = {r_squared:.6f})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=100)
    print(f"Plot saved to {output_file}")
    plt.show()


if __name__ == "__main__":
    # Generate data
    x, y = generate_data()
    df = pd.DataFrame({'x': x, 'y': y})
    
    # Save data
    save_data(df)
    
    # Load data
    df_loaded = load_data()
    
    # Fit model
    poly, model, r_squared, X, y_data = fit_data(df_loaded, 3)
    
    # Plot results
    plot_results(df_loaded, poly, model, r_squared)
