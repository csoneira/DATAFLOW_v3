if n_study_fit:
        
        print("----------------------------------------------------------------------")
        print("------------------------ The n study fit -----------------------------")
        print("----------------------------------------------------------------------")
        
        first_approach = True
        
        if first_approach:
            # Fit the histogram of df["theta"] values per each definitive_tt to the a function which is:
            # F_ij(theta, phi) = A_ij * cos(theta)^n_ij * R_ij(theta, phi)
            # where R_ij is the response function of a telescope made of square planes of 300 mm
            # length and at a distance of abs(z_position[i] - z_positions[j]). First simulate the R_ij for
            # the cases 13, 24, 14, 23. Note that definitive_tt = 123 has the same response as definitive_tt
            # = 13, because what count are the top and bottom planes in each case. Simulate and plot the
            # R_ij for those cases, then fit the theta to F_ij, and plot it. Obtain an A_ij and n_ij
            # per each case. Then sum all to obtain a realistic total flux crossing the detector. Go!
            
            use_only_theta = True
            if use_only_theta:
                
                
                # STEP 1: response function calculation

                # Simulation parameters
                N = 3000000  # number of events per configuration
                PLANE_SIZE = 300  # mm, square plane side
                HALF_SIZE = PLANE_SIZE / 2

                pairs = {
                    '13': (1, 3),
                    '24': (2, 4),
                    '14': (1, 4),
                    '23': (2, 3),
                    
                    '123': (1, 3),
                    '234': (2, 4),
                    '1234': (1, 4),
                }

                def simulate_telescope_response(z1, z2, N=N):
                    """Simulate direction distribution of tracks crossing square planes at z1 and z2."""
                    # Random (x, y) points on both planes
                    x1 = np.random.uniform(-HALF_SIZE, HALF_SIZE, N)
                    y1 = np.random.uniform(-HALF_SIZE, HALF_SIZE, N)
                    x2 = np.random.uniform(-HALF_SIZE, HALF_SIZE, N)
                    y2 = np.random.uniform(-HALF_SIZE, HALF_SIZE, N)

                    dx = x2 - x1
                    dy = y2 - y1
                    dz = z2 - z1

                    r = np.sqrt(dx**2 + dy**2 + dz**2)
                    theta = np.arccos(np.abs(dz) / r)
                    phi = np.arctan2(dy, dx)

                    return theta, phi

                # Simulate and histogram theta density
                theta_bins = np.linspace(0, np.pi / 2, 100)
                theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
                response_histograms = {}

                for label, (i, j) in pairs.items():
                    z1, z2 = z_positions[i-1], z_positions[j-1]
                    theta, phi = simulate_telescope_response(z1, z2)
                    hist, _ = np.histogram(theta, bins=theta_bins, density=True)
                    response_histograms[label] = hist

                # Plot simulated response densities
                fig, ax = plt.subplots(figsize=(8, 6))
                for label in response_histograms:
                    ax.plot(theta_centers * 180 / np.pi, response_histograms[label], label=f'{label}')
                ax.set_xlabel(r'$\theta$ [deg]')
                ax.set_ylabel(r'Normalized density')
                ax.set_title('Simulated Angular Response $R_{ij}(\\theta)$ from Geometry')
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                if save_plots:
                    final_filename = f'{fig_idx}_simulated_response_function.png'
                    fig_idx += 1
                    save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                    plot_list.append(save_fig_path)
                    plt.savefig(save_fig_path, format='png')
                if show_plots:
                    plt.show()
                plt.close()


                # STEP 2: fit the theta to F_ij, and plot it

                # Assuming df is available and contains 'theta' and 'definitive_tt'

                
                
                # Reuse the existing response_histograms from earlier simulation
                fit_results_real = {}

                # Loop over response functions and fit real data per definitive_tt
                for label, response in response_histograms.items():
                    if label not in df['definitive_tt'].astype(str).unique():
                        continue

                    # Extract real theta data for this topology
                    df_tt = df[df['definitive_tt'].astype(str) == label]
                    theta_data = df_tt['theta'].dropna().values
                    if len(theta_data) < 20:
                        continue

                    # Histogram real data
                    data_hist, _ = np.histogram(theta_data, bins=theta_bins, density=False)

                    # Create interpolated response function
                    R_interp = interp1d(theta_centers, response, kind='linear', bounds_error=False, fill_value=0)

                    # Fit function: F(theta) = A * cos(theta)^n * R_ij(theta)
                    def fit_func(theta, A, n, B, C):
                        return (A * np.cos(theta) ** 2 + B * np.cos(theta) ** n + C) * R_interp(theta)

                    try:
                        popt, _ = curve_fit(fit_func, theta_centers, data_hist,\
                            p0=[1.0, 2.0, 1.0, 1.0], bounds=([0, 0, 0, 0], [np.inf, 100, np.inf, np.inf]))
                        A_fit, n_fit, B_fit, C_fit = popt
                        fit_results_real[label] = (A_fit, n_fit, B_fit, C_fit)

                        # Plot
                        fig, ax = plt.subplots(figsize=(7, 5))
                        ax.plot(theta_centers * 180 / np.pi, data_hist, label='Data', color='black')
                        ax.plot(theta_centers * 180 / np.pi, fit_func(theta_centers, *popt), '--', \
                            label=f'Fit: A={A_fit:.2f}, n={n_fit:.2f}, B={B_fit:.2f}, C={C_fit:.2f}', color='red')
                        ax.set_xlabel(r'$\theta$ [deg]')
                        ax.set_ylabel('Normalized density')
                        ax.set_title(f'Fit of $F_{{{label}}}(\\theta)$ to Real Data')
                        ax.legend()
                        ax.grid(True)
                        plt.tight_layout()
                        plt.show()
                    except RuntimeError:
                        print(f"Fit failed for {label}")

                fit_results_real

            else:
                

                # Define the simulation again for theta and phi
                N = 1000000
                PLANE_SIZE = 300  # mm
                HALF_SIZE = PLANE_SIZE / 2

                pairs = {
                    '13': (1, 3),
                    '24': (2, 4),
                    '14': (1, 4),
                    '23': (2, 3),
                    
                    '123': (1, 3),
                    '234': (2, 4),
                    '1234': (1, 4),
                }

                

                # Set z_positions for the 4 planes (in mm)
                
                # Reuse definitions
                PLANE_SIZE = 300  # mm
                HALF_SIZE = PLANE_SIZE / 2
                N = 2000000

                # Define angular binning
                theta_bins = np.linspace(0, np.pi / 2.5, 20)
                phi_bins = np.linspace(-np.pi, np.pi, 20)
                theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
                phi_centers = 0.5 * (phi_bins[:-1] + phi_bins[1:])
                THETA_MESH, PHI_MESH = np.meshgrid(theta_centers, phi_centers, indexing='ij')


                # Simulate response in 2D (theta, phi)
                def simulate_theta_phi(z1, z2, N=N):
                    x1 = np.random.uniform(-HALF_SIZE, HALF_SIZE, N)
                    y1 = np.random.uniform(-HALF_SIZE, HALF_SIZE, N)
                    x2 = np.random.uniform(-HALF_SIZE, HALF_SIZE, N)
                    y2 = np.random.uniform(-HALF_SIZE, HALF_SIZE, N)
                
                    dx = x2 - x1
                    dy = y2 - y1
                    dz = z2 - z1
                    r = np.sqrt(dx**2 + dy**2 + dz**2)
                    theta = np.arccos(np.abs(dz) / r)
                    phi = np.arctan2(dy, dx)
                    return theta, phi

                # Simulate and construct 2D histograms
                response_2d = {}
                response_functions_2d = {}
                for label, (i, j) in pairs.items():
                    z1, z2 = z_positions[i-1], z_positions[j-1]
                    theta_sim, phi_sim = simulate_theta_phi(z1, z2)
                    H, _, _ = np.histogram2d(theta_sim, phi_sim, bins=[theta_bins, phi_bins], density=True)
                    response_2d[label] = H
                    response_functions_2d[label] = H

                # Plot simulated 2D response R_ij(theta, phi)
                fig, axes = plt.subplots(2, 4, subplot_kw={'projection': 'polar'}, figsize=(18, 10))
                axes = axes.flatten()
                for idx, (label, R) in enumerate(response_2d.items()):
                    ax = axes[idx]
                    pcm = ax.pcolormesh(PHI_MESH, THETA_MESH, R, shading='auto', cmap='viridis')
                    ax.set_title(f'Response R_{{{label}}}($\\theta$, $\\phi$)')
                    fig.colorbar(pcm, ax=ax, orientation='vertical', pad=0.1)

                for ax in axes[len(response_2d):]:
                    ax.axis('off')

                plt.suptitle('Simulated 2D Response Functions $R_{ij}(\\theta, \\phi)$', fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.show()


                # Now proceed to fitting with updated response_functions_2d
                fit_results = {}

                for label, interp_R in {
                    l: RegularGridInterpolator((theta_centers, phi_centers), response_functions_2d[l], bounds_error=False, fill_value=0)
                    for l in response_functions_2d
                }.items():
                    if label not in df['definitive_tt'].astype(str).unique():
                        continue

                    df_tt = df[df['definitive_tt'].astype(str) == label]
                    theta_data = df_tt['theta'].dropna().values
                    phi_data = df_tt['phi'].dropna().values

                    if len(theta_data) < 50:
                        continue

                    data_hist, _, _ = np.histogram2d(theta_data, phi_data, bins=[theta_bins, phi_bins], density=False)
                    theta_phi_coords = np.array(np.meshgrid(theta_centers, phi_centers, indexing='ij'))
                    theta_phi_flat = np.vstack([theta_phi_coords[0].ravel(), theta_phi_coords[1].ravel()]).T
                    R_vals = interp_R(theta_phi_flat).reshape(len(theta_centers), len(phi_centers))

                    def fit_func(grid_flat, A, n, B):
                        theta_flat = grid_flat[:, 0]
                        phi_flat = grid_flat[:, 1]
                        R_eval = interp_R(np.vstack([theta_flat, phi_flat]).T)
                        return (A * np.cos(theta_flat) ** n + B) * R_eval

                    data_flat = data_hist.ravel()
                    coords_flat = theta_phi_flat
                    mask = data_flat > 0

                    try:
                        popt, _ = curve_fit(fit_func, coords_flat[mask], data_flat[mask], p0=[1.0, 2.0, 1.0], bounds=([0, 0, 0], [np.inf, 100, np.inf]))
                        fit_results[label] = popt

                        fit_vals = fit_func(coords_flat, *popt).reshape(data_hist.shape)

                        # Plotting
                        fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(14, 6))
                        PHI_MESH, THETA_MESH = np.meshgrid(phi_centers, theta_centers, indexing='ij')

                        pcm1 = axes[0].pcolormesh(PHI_MESH, THETA_MESH, data_hist.T, shading='auto', cmap='viridis')
                        axes[0].set_title(f'Data: {label}')
                        fig.colorbar(pcm1, ax=axes[0])

                        pcm2 = axes[1].pcolormesh(PHI_MESH, THETA_MESH, fit_vals.T, shading='auto', cmap='viridis')
                        axes[1].set_title(f'Fit: A={popt[0]:.2f}, n={popt[1]:.2f}, B={popt[2]:.2f}')
                        fig.colorbar(pcm2, ax=axes[1])

                        plt.tight_layout()
                        plt.show()

                    except RuntimeError:
                        fit_results[label] = None
        
        else:

            # Okey, so look out: i want you to change this to take a slightly different approach: instead of 
            # calculating the response functions just considerng two planes and thats all, you are going to 
            # do the following: generate tracks in a uniform distribution in zenith and azimuth and in a plane 
            # above the first plane of the detector (z_positions[0]) but from -1000 to 1000, then you are going 
            # to check which planes the track crosses, according to its theta, phi, x, y and the z_positions of 
            # the planes, which are in x and y between -150 and 150. Then you are going to apply a efficiency, 
            # which will be 0.95 * theta**0.001, to determine if the plane is detected or not for a certain plane.
            # So in the end each track will have a crossing_tt and a measured_tt (when applying the efficiency 
            # over crossing_tt), you calculate, from the total tracks in a bin of theta, how many where detected 
            # in each combination of planes, so you get a response function but taking account efficiencies. So 
            # in this case, 123 is not the same as 13, because for an event to be 13 then it must NOT be detected in plane 2,

            

            # Simulation parameters
            N = 50_000_000
            PLANE_SIZE = 300  # mm, square plane side
            HALF_SIZE = PLANE_SIZE / 2
            EFFICIENCY_FUNC = lambda theta: 0.93 + 0.02 * theta ** 0.9  # simplified efficiency

            # Detector configuration
            n_planes = len(z_positions)

            # Define angular and spatial domain of generation
            z_gen = 0  # generation plane
            x_gen = np.random.uniform(-HALF_SIZE * 5, HALF_SIZE * 5, N)
            y_gen = np.random.uniform(-HALF_SIZE * 5, HALF_SIZE * 5, N)

            n = 2
            u = np.random.uniform(0, 1, N)
            theta_gen = np.arccos((1 - u) ** (1 / (n + 1)))
            # theta_gen = np.arccos(np.random.uniform(0, 1, N))  # uniform in cosθ

            phi_gen = np.random.uniform(-np.pi, np.pi, N)

            # Compute direction vector components
            dx = np.sin(theta_gen) * np.cos(phi_gen)
            dy = np.sin(theta_gen) * np.sin(phi_gen)
            dz = np.cos(theta_gen)

            # Track intersection with each plane
            crossing_tt = []
            measured_tt = []

            for i_plane, z_det in enumerate(z_positions):
                t = (z_det - z_gen) / dz
                x_det = x_gen + t * dx
                y_det = y_gen + t * dy
                inside = (np.abs(x_det) <= HALF_SIZE) & (np.abs(y_det) <= HALF_SIZE)
                if i_plane == 0:
                    crossing_mask = inside.astype(int)
                else:
                    crossing_mask = np.vstack([crossing_mask, inside.astype(int)])

            # Transpose to get shape (N, n_planes)
            crossing_mask = crossing_mask.T

            for i in range(N):
                # print a message each 100000 events
                if i % 100000 == 0:
                    print(f"Processing event {i} of {N}... {i/N*100:.1f} % complete")
                
                crossing_planes = [str(j + 1) for j in range(n_planes) if crossing_mask[i, j]]
                measured_planes = []
                for j in range(n_planes):
                    if crossing_mask[i, j] and np.random.rand() < EFFICIENCY_FUNC(theta_gen[i]):
                        measured_planes.append(str(j + 1))
                if crossing_planes:
                    crossing_tt.append(''.join(crossing_planes))
                else:
                    crossing_tt.append('0')
                if measured_planes:
                    measured_tt.append(''.join(measured_planes))
                else:
                    measured_tt.append('0')

            # Store by configuration
            theta_bins = np.linspace(0, np.pi / 2, 50)
            theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
            response_histograms = defaultdict(lambda: np.zeros(len(theta_centers)))

            # Build histograms for each measured_tt
            for i in range(N):
                label = measured_tt[i]
                if label != '0':
                    bin_idx = np.digitize(theta_gen[i], theta_bins) - 1
                    if 0 <= bin_idx < len(theta_centers):
                        response_histograms[label][bin_idx] += 1

            # Normalize each histogram
            for label in response_histograms:
                total = np.sum(response_histograms[label])
                if total > 0:
                    response_histograms[label] /= total


            # Convert response_histograms to a DataFrame for saving as CSV
            response_df = pd.DataFrame.from_dict(response_histograms, orient='index', columns=theta_centers)
            response_df.to_csv(os.path.join(base_directories["figure_directory"], 'response_histograms.csv'))


            read_response = True
            if read_response:
                # Load response_histograms from the csv file
                response_df = pd.read_csv(os.path.join(base_directories["figure_directory"], 'response_histograms.csv'), index_col=0)



            # Plot simulated response densities

            fig, ax = plt.subplots(figsize=(8, 6))
            colors = cm.viridis(np.linspace(0, 1, len(response_histograms)))
            for idx, label in enumerate(sorted(response_histograms.keys())):
                if label in ['0', '', '1', '2', '3', '4', '14']:
                    continue
                ax.plot(theta_centers * 180 / np.pi, response_histograms[label], label=label, color=colors[idx])

            ax.set_xlabel(r'$\theta$ [deg]')
            ax.set_ylabel(r'Normalized density')
            ax.set_title('Simulated Angular Response $R_{tt}(\\theta)$ with Efficiency and Plane Logic')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.tight_layout()
            if save_plots:
                final_filename = f'{fig_idx}_sim_ang_fit_response_{label}.png'
                fig_idx += 1
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')
            if show_plots:
                plt.show()
            plt.close()


            # STEP 2: Fit the theta distribution to F_ij(theta) model using the simulated R_ij

            # Store fit results
            fit_results_real = {}

            # Perform fitting
            for label, response in response_histograms.items():
                if label not in df['definitive_tt'].astype(str).unique():
                    continue

                df_tt = df[df['definitive_tt'].astype(str) == label]
                theta_data = df_tt['theta'].dropna().values
                if len(theta_data) < 30:
                    continue

                data_hist, _ = np.histogram(theta_data, bins=theta_bins, density=False)
                R_interp = interp1d(theta_centers, response, kind='linear', bounds_error=False, fill_value=0)
                
                # Fit model: (A * cos² + B * cos^n + C) * R(θ)
                def fit_func(theta, A, n, B, C):
                    cos_t = np.cos(theta)
                    sin_t = np.sin(theta)
                    # return (A * cos_t**2 + B * cos_t**n + C) * R_interp(theta)
                    # return (A * cos_t**n + B) * R_interp(theta) * sin_t
                    # return (A * cos_t**n) * R_interp(theta) * sin_t
                    return (A * cos_t**n) * R_interp(theta)

                try:
                    popt, _ = curve_fit(fit_func, theta_centers, data_hist, p0=[1.0, 2.0, 1.0, 1.0],
                                        bounds=([0, 0, 0, 0], [np.inf, 100, np.inf, np.inf]))
                    fit_results_real[label] = (popt, data_hist, response)
                except RuntimeError:
                    fit_results_real[label] = None

            # Plotting results in subplots
            n = len(fit_results_real)
            ncols = 3
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

            for idx, (label, result) in enumerate(fit_results_real.items()):
                row, col = divmod(idx, ncols)
                ax = axes[row][col]
                if result is None:
                    ax.set_visible(False)
                    continue

                popt, data_hist, response = result
                R_interp = interp1d(theta_centers, response, kind='linear', bounds_error=False, fill_value=0)

                def fit_func(theta, A, n, B, C):
                    cos_t = np.cos(theta)
                    return (A * cos_t**n) * R_interp(theta)

                ax.plot(theta_centers * 180 / np.pi, data_hist, label='Data', color='black')
                ax.plot(theta_centers * 180 / np.pi, fit_func(theta_centers, *popt), '--', color='red',
                        label=fr'Fit: $A$={popt[0]:.2f}, $n$={popt[1]:.2f}')
                ax.set_xlabel(r'$\theta$ [deg]')
                ax.set_ylabel('Counts per bin')
                ax.set_title(f'Topology {label}')
                ax.legend()
                ax.grid(True)

            # Hide unused axes
            for i in range(n, nrows * ncols):
                row, col = divmod(i, ncols)
                axes[row][col].set_visible(False)

            plt.tight_layout()
            if save_plots:
                final_filename = f'{fig_idx}_fit_response_all.png'
                fig_idx += 1
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')
            if show_plots:
                plt.show()
            plt.close()
