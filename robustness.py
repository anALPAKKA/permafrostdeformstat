def bootstrap_ks_test(data1, data2, uncertainty, min_sample_size=3000, num_bootstrap=1000, alpha=0.05):
    """
    Perform a bootstrap K-S test to assess if two distributions differ significantly,
    accounting for measurement uncertainty and internal variability.
    
    Parameters:
    - data1: array-like, first dataset
    - data2: array-like, second dataset
    - uncertainty: float, measurement uncertainty (e.g., standard deviation of the noise)
    - min_sample_size: int, minimum number of data points in each resampled dataset
    - num_bootstrap: int, number of bootstrap iterations
    - alpha: float, significance level (default is 0.05)
    
    Returns:
    - p_value_mean: mean p-value over all bootstrap iterations
    - significant_diffs: fraction of bootstrap iterations that show significant difference
    """
    p_values = []
    for i in range(num_bootstrap):
        # Print the iteration number every 10 iterations
        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1} / {num_bootstrap}")
        
        # Ensure each dataset has at least min_sample_size points by resampling with replacement if necessary
        if len(data1) < min_sample_size:
            sampled_data1 = np.random.choice(data1, min_sample_size, replace=True)
        else:
            sampled_data1 = data1 + np.random.normal(0, uncertainty, len(data1))
        
        if len(data2) < min_sample_size:
            sampled_data2 = np.random.choice(data2, min_sample_size, replace=True)
        else:
            sampled_data2 = data2 + np.random.normal(0, uncertainty, len(data2))
        
        # Add noise to simulate measurement uncertainty
        noisy_data1 = sampled_data1 + np.random.normal(0, uncertainty, len(sampled_data1))
        noisy_data2 = sampled_data2 + np.random.normal(0, uncertainty, len(sampled_data2))
        
        # Perform K-S test on the noisy data
        _, p_value = ks_2samp(noisy_data1, noisy_data2)
        p_values.append(p_value)
    
    # Calculate mean p-value and fraction of significant differences
    p_value_mean = np.mean(p_values)
    significant_diffs = np.mean(np.array(p_values) < alpha)
    
    print("Mean P-value:", p_value_mean)
    print("Fraction of significant results:", significant_diffs)
    
    return p_value_mean, significant_diffs



data1 = prepare_subsidence_data('anktv','201516', 'orgsub')[0].ravel()
data2 = prepare_subsidence_data('anktv','201617', 'orgsub')[0].ravel()
data3 = prepare_subsidence_data('anktv','201718', 'orgsub')[0].ravel()
merged_data_anktv = np.concatenate((data1, data2,data3))
data1 = prepare_subsidence_data('ykn','201516', 'orgsub')[0].ravel()
data2 = prepare_subsidence_data('ykn','201617', 'orgsub')[0].ravel()
data3 = prepare_subsidence_data('ykn','201718', 'orgsub')[0].ravel()
merged_data_ykn = np.concatenate((data1, data2,data3))
data1 = prepare_subsidence_data('ntk','201516', 'orgsub')[0].ravel()
data2 = prepare_subsidence_data('ntk','201617', 'orgsub')[0].ravel()
data3 = prepare_subsidence_data('ntk','201718', 'orgsub')[0].ravel()
merged_data_ntk = np.concatenate((data1, data2,data3))
data1 = prepare_subsidence_data('invk','201516', 'orgsub')[0].ravel()
data2 = prepare_subsidence_data('invk','201617', 'orgsub')[0].ravel()
data3 = prepare_subsidence_data('invk','201718', 'orgsub')[0].ravel()
merged_data_invk = np.concatenate((data1, data2,data3))


# Subsampling to reduce dataset size
sample_size = 5000  # Adjust based on what your system can handle

# Sample from each merged dataset
sample_data_anktv = np.random.choice(merged_data_anktv, sample_size, replace=False)
sample_data_ykn = np.random.choice(merged_data_ykn, sample_size, replace=False)
sample_data_ntk = np.random.choice(merged_data_ntk, sample_size, replace=False)
sample_data_invk = np.random.choice(merged_data_invk, sample_size, replace=False)

# Run the bootstrap K-S test on the sampled data
bootstrap_ks_test(sample_data_anktv, sample_data_ykn, uncertainty=0.01, min_sample_size=3000, num_bootstrap=10, alpha=0.05)
bootstrap_ks_test(sample_data_anktv, sample_data_ntk, uncertainty=0.01, min_sample_size=3000, num_bootstrap=10, alpha=0.05)
bootstrap_ks_test(sample_data_ykn, sample_data_ntk, uncertainty=0.01, min_sample_size=3000, num_bootstrap=10, alpha=0.05)
