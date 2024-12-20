import os
import numpy as np
import subprocess
import nibabel as nib
import numpy as np
from nilearn.input_data import NiftiLabelsMasker

# Utils functions

# Utility for get the output subject file name
def compute_file_name(subject, atlas_name):
    words = subject.split('_')
    words[1] = "00" + words[1]
    return f'{"_".join(words)}_rois_{atlas_name}.1D' 

# Utility for get the original anatomical image from a subject
def original_anat_image(data_path, subject):
    return os.path.join(data_path, subject, "anat/NIfTI/mprage.nii.gz")

# Utility for get the original functional image from a subject
def original_func_image(data_path, subject):
    return os.path.join(data_path, subject, "rest/NIfTI/rest.nii.gz")

# Utility function to create directories
def create_directory(path, description=None):
    if not os.path.exists(path):
        run_command(f'mkdir -p {path}', description)

# Utility function to copy files
def copy_file(src, dest, description=None):
    run_command(f'cp {src} {dest}', description)

# Compute polynomial regressors
def compute_polynomial_regressors(num_timepoints):
    linear = np.linspace(-1, 1, num_timepoints)  # Linear trend
    quadratic = np.linspace(-1, 1, num_timepoints) ** 2  # Quadratic trend
    return linear, quadratic

# Utility function to run shell commands
def run_command(command, description=None):
    if description:
        print(description)
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        exit(1)
    print(result.stdout)
    return result.stdout

# Anatomical preprocessing
def anat_prep(base_dir, data_path, subject):
    print("Starting anatomical preprocessing...")
   
    # Define paths
    subject_anat_dir = os.path.join(base_dir, subject, "anat")
    brain_extraction_dir = os.path.join(subject_anat_dir, "brain_extraction")
    segmentation_tissue_dir = os.path.join(subject_anat_dir, "segmentation_tissue")
    registration_dir = os.path.join(subject_anat_dir, "registration")
    MNI_standard = "$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz"
    
    # Create directories
    create_directory(subject_anat_dir, "Creating anatomical folder...")
    create_directory(brain_extraction_dir, "Creating brain extraction folder...")
    create_directory(segmentation_tissue_dir, "Creating segmentation tissue folder...")
    create_directory(registration_dir, "Creating registration folder...") 

    anat_image = original_anat_image(data_path, subject)
    
    # Brain extraction
    brain_extraction_output = os.path.join(brain_extraction_dir, "brain_extraction")

    bet_command = f'bet {anat_image} {brain_extraction_output} -o -m -s -f 0.4 -R'
    run_command(bet_command, "Performing brain extraction...")
    brain_extraction_image = f"{brain_extraction_output}.nii.gz"

    # Copy brain extraction image to segmentation tissue folder
    dest_brain_extraction_image = os.path.join(segmentation_tissue_dir, "brain_extraction.nii.gz")
    copy_file(brain_extraction_image, dest_brain_extraction_image, "Copying brain extraction image...")
    
    # Compute the affine registration matrix
    registration_matrix = os.path.join(registration_dir, "registration_matrix.mat")
    compute_matrix_command = f'flirt -in {dest_brain_extraction_image} -ref {MNI_standard} -out {os.path.join(registration_dir, "anat_to_standard")} -omat {registration_matrix}'
    
    run_command(compute_matrix_command, "Computing affine registration matrix...")

    # Perform tissue segmentation
    fast_command = (
        f'fast -A {registration_matrix} -B {dest_brain_extraction_image}'
    )
    run_command(fast_command, "Performing tissue segmentation...")
    
    # Compute tissue masks
    segmentation_tissue_output = os.path.join(segmentation_tissue_dir, "brain_extraction")
    
    for i in range(3):  # Generate masks for tissue types 0, 1, 2
        pve_file = f"{segmentation_tissue_output}_pve_{i}.nii.gz"
        mask_file = f"{segmentation_tissue_output}_pve_{i}_mask.nii.gz"
        fslmaths_command = f"fslmaths {pve_file} -bin {mask_file}"
        run_command(fslmaths_command, f"Computing tissue mask for PVE {i}...")

    print("Anatomical preprocessing completed successfully.")

# Functional preprocessing
def drop_first_slices(dir, image):
  output = os.path.join(dir, "drop_slices")
  command = f'fslroi {image} {output} 4 -1'

  run_command(command, "Drop first 4 volumes...")

  return f'{output}.nii.gz'
  
def slice_timing_correction(dir, image):
  # Params
  TR = 2000
  direction = 3
  
  output = os.path.join(dir, "slice_timing_correction")
  command = f'slicetimer -i {image} -o {output} -r {TR} -d {direction} --odd'
  
  run_command(command,  "Slice timing correction...")
  return f'{output}.nii.gz'

def motion_correction(dir, image): 
  output = os.path.join(dir, "motion_correction")
  command = f'mcflirt -in {image} -out {output} -mats -plots -dof 12 -meanvol'
  
  run_command(command, "Motion correction...")
  
  return f'{output}.nii.gz'

def brain_extraction(dir, image):
  output = os.path.join(dir, "brain_extraction")
  command = f'bet {image} {output} -o -m -s -f 0.3 -R -F'
  
  run_command(command, "Brain extraction...")
  return f'{output}.nii.gz'
    
def intensity_normalization(dir, image):    
  output = os.path.join(dir, "intensity_normalization") 
  command = f'fslmaths {image} -ing 1000 {output}'
  
  run_command(command, "Intensity normalization...")
  return f'{output}.nii.gz'

def motion_regression(dir, image, motion_correction_dir):
  # Load motion parameters
  motion_params = np.loadtxt(os.path.join(motion_correction_dir, 'motion_correction.par'))

  # Generate lagged motion parameters
  lagged_motion_params = np.roll(motion_params, shift=1, axis=0)
  lagged_motion_params[0, :] = 0  # Avoid circular shift

  # Compute squared terms
  squared_motion_params = motion_params ** 2
  squared_lagged_motion_params = lagged_motion_params ** 2

  # Combine all regressors
  nuisance_regressors = np.hstack([motion_params, lagged_motion_params, squared_motion_params, squared_lagged_motion_params])

  # Save combined regressors
  motion_regressors = os.path.join(dir, 'motion_regressors.txt')
  np.savetxt(motion_regressors, nuisance_regressors, fmt="%.6f")

  # Apply nuisance regression 
  output = os.path.join(dir, 'cleaned_motion_image')
  nuisance_regression_command = f'fsl_regfilt -i {image} -d {motion_regressors} -f "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24" -o {output}'

  run_command(nuisance_regression_command, "Performing motion nuisance regression...")

  return f'{output}.nii.gz'

def transform_tissue_mask_to_func(dir, subject_anat_dir, subject_anat_image, image):  
    # Compute transformation matrix
    output = os.path.join(dir, 'func2anat')
    transformation_matrix = os.path.join(dir, 'func2anat_matrix.mat')
    transformation_matrix_command = f'flirt -in {image} -ref {subject_anat_image} -omat {transformation_matrix} -out {output}'
   
    run_command(transformation_matrix_command, "Computing affine registration matrix...")

    # Inverse the transformation matrix
    inverse_transformation_matrix = os.path.join(dir, 'anat2func_matrix.mat')
    inverse_transformation_matrix_command = f'convert_xfm -omat {inverse_transformation_matrix} -inverse {transformation_matrix}'
    
    run_command(inverse_transformation_matrix_command, "Computing inverse transformation matrix...")
    
    # Transform tissue masks to functional space
    # White matter (2)
    output_wm = os.path.join(dir, 'wm_mask_func')
    transformation_wm_mask_command = f'flirt -in {os.path.join(subject_anat_dir, "segmentation_tissue", "brain_extraction_pve_2_mask.nii.gz")} -ref {image} -applyxfm -init {inverse_transformation_matrix} -out {output_wm} -interp nearestneighbour'

    run_command(transformation_wm_mask_command, "Transform WM mask from anatomical to functional")
    
    # CSF (0)
    output_csf = os.path.join(dir, 'csf_mask_func')
    transformation_csf_mask_command = f'flirt -in {os.path.join(subject_anat_dir, "segmentation_tissue", "brain_extraction_pve_0_mask.nii.gz")} -ref {image} -applyxfm -init {inverse_transformation_matrix} -out {output_csf} -interp nearestneighbour'
    
    run_command(transformation_csf_mask_command, "Transform CSF mask from anatomical to functional")
    
    return output_wm, output_csf

def tissue_regression(dir, subject_anat_dir, subject_anat_image, image):
    # Transform wm and csf tissue mask to functional space 
    wm_mask, csf_mask = transform_tissue_mask_to_func(dir, subject_anat_dir, subject_anat_image, image)  

    # Extract tissue signals
    # White matter
    output_wm = os.path.join(dir, "wm_signal.txt")
    extract_wm_signals_command = f'fslmeants -i {image} -m {wm_mask} -o {output_wm}'

    run_command(extract_wm_signals_command, "Extract WM signal from functional image...")
   
    # Extract CSF signals
    output_csf = os.path.join(dir, "csf_signal.txt")
    extract_csf_signals_command = f'fslmeants -i {image} -m {csf_mask} -o {output_csf}'

    run_command(extract_csf_signals_command, "Extract CSF signal from functional image...")
    
    output_tissue_regressors = os.path.join(dir, 'tissue_regressors.txt') 
    save_command = f'paste {output_wm} {output_csf} > {output_tissue_regressors}'

    run_command(save_command, "Combine CSF and WM regressors")

    # Apply nuisance regression 
    output = os.path.join(dir, 'cleaned_tissue_image')
    nuisance_regression_command = f'fsl_regfilt -i {image} -d {output_tissue_regressors} -f "1,2" -o {output}'

    run_command(nuisance_regression_command, "Performing tissue nuisance regression...")

    return f'{output}.nii.gz'

def polynomial_regression(dir, image):
    # Define the polynomial regressors 
    get_num_time_points_command = f'fslnvols {image}'
        
    num_time_points = int(run_command(get_num_time_points_command, "Get the number of time points...").strip())
    
    # Compute linear and quadratic regressors
    linear_reg, quadratic_reg = compute_polynomial_regressors(num_time_points)

    poly_trends = os.path.join(dir, "poly_trends.txt")
    np.savetxt(poly_trends, np.column_stack((linear_reg, quadratic_reg)), fmt='%.5f')

    # Apply nuisance regression 
    output = os.path.join(dir, 'cleaned_poly_trends_image')
    nuisance_regression_command = f'fsl_regfilt -i {image} -d {poly_trends} -f "1,2" -o {output}'

    run_command(nuisance_regression_command, "Performing polynomial trends regression...")

    return f'{output}.nii.gz'

def nuisance_reggresion(dir, image, motion_correction_dir, subject_anat_dir, subject_anat_image):
    cleaned_motion_image = motion_regression(dir, image, motion_correction_dir)

    cleaned_tissue_image = tissue_regression(dir, subject_anat_dir, subject_anat_image, cleaned_motion_image)

    cleaned_image = polynomial_regression(dir, cleaned_tissue_image)

    return cleaned_image

def band_pass_filter(dir, image):  
    output_mean = os.path.join(dir, "func_mean")
    command = f'fslmaths {image} -Tmean {output_mean}'
  
    run_command(command, "Compute mean functional image...")

    # Params
    tr = 2.0  # Repetition time in seconds
    highpass_cutoff = 0.01  # Highpass frequency cutoff in Hz
    lowpass_cutoff = 0.1  # Lowpass frequency cutoff in Hz
    
    # Calculate sigma values
    highpass_sigma = 1 / (2 * tr * highpass_cutoff) if highpass_cutoff > 0 else -1
    lowpass_sigma = 1 / (18 * tr * lowpass_cutoff) if lowpass_cutoff > 0 else -1
    
    output = os.path.join(dir, "filter")
    command = f'fslmaths {image} -bptf {highpass_sigma} {lowpass_sigma} {output}'
  
    run_command(command, "Band pass filter...")
    return f'{output}.nii.gz'

def anatomical_registration(dir, image, subject_anat_image):
    output = os.path.join(dir, "func2anat")
    transformation_matrix = os.path.join(dir, 'func2anat_matrix.mat')
    command = f'flirt -in {image} -ref {subject_anat_image} -out {output} -omat {transformation_matrix} -cost normmi -dof 12 -interp sinc'
  
    run_command(command, "Compute transformation matrix from functional to anatomical...")

    # Apply transformation to 4D functional image
    output = os.path.join(dir, "func2anat_4d")
    command = f'flirt -in {image} -ref {subject_anat_image} -applyxfm -init {transformation_matrix} -out {output}'
   
    run_command(command, "Apply transformation to 4D functional image...")

    return f'{output}.nii.gz'

def standard_registration(dir, subject_anat_dir, image, MNI_standard):
  transformation_matrix = os.path.join(subject_anat_dir, 'registration', 'registration_matrix.mat')
 
  # Apply transformation to 4D functional image
  output = os.path.join(dir, "func2standard_4d")
  command = f'flirt -in {image} -ref {MNI_standard} -applyxfm -init {transformation_matrix} -out {output}'
  
  run_command(command, "Apply standard transformation to 4D functional image...")

  return f'{output}.nii.gz'

def rois_extraction(subject, image, atlas_name):
    file_name = compute_file_name(subject, atlas_name)
    output = os.path.join("output", f'rois_{atlas_name}', file_name)
    atlas = os.path.join("atlas", atlas_name, f"{atlas_name}_roi_atlas_in_standard.nii.gz") 
    
    # Load images
    functional_image = nib.load(image)
    atlas_image = nib.load(atlas)

    # Initialize the masker
    masker = NiftiLabelsMasker(
        labels_img=atlas_image,
        standardize=False,  # Keep raw signals
        detrend=False       # Keep original trends
    )

    # Extract mean time series for each ROI
    mean_time_series = masker.fit_transform(functional_image)

    # Save the result
    np.savetxt(output, mean_time_series, delimiter="\t")
    print("Shape of mean time series:", mean_time_series.shape)
    print(f"Mean time series saved to {output}'")

    return output

def func_prep(base_dir, data_path, subject, atlas_name):
    print("Starting functional preprocessing...")
   
    # Define paths
    subject_func_dir = os.path.join(base_dir, subject, "func")
    drop_first_slices_dir = os.path.join(subject_func_dir, "drop_first_slices")
    slice_timing_correction_dir = os.path.join(subject_func_dir, "slice_timing_correction")
    motion_correction_dir = os.path.join(subject_func_dir, 'motion_correction')
    brain_extraction_dir = os.path.join(subject_func_dir, 'brain_extraction')
    intensity_normalization_dir = os.path.join(subject_func_dir, 'intensity_normalization')
    nuisance_reggresion_dir = os.path.join(subject_func_dir, 'nuisance_reggresion')
    band_pass_filter_dir = os.path.join(subject_func_dir, 'filter')
    anatomical_registration_dir = os.path.join(subject_func_dir, "anatomical_registration")
    standard_registration_dir = os.path.join(subject_func_dir, 'standard_registration')
    
    MNI_standard = "$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz"
    
    # Create directories
    create_directory(subject_func_dir, "Creating functional folder...")
    create_directory(drop_first_slices_dir, "Creating drop first slices folder...")
    create_directory(slice_timing_correction_dir, "Creating slice timing correction folder...")
    create_directory(motion_correction_dir, "Creating motion correction folder...")
    create_directory(brain_extraction_dir, "Creating brain extraction folder...")
    create_directory(intensity_normalization_dir, "Creating intensity normalization folder...")
    create_directory(nuisance_reggresion_dir, "Creating nuisance reggresion folder...")
    create_directory(band_pass_filter_dir, "Creating band pass filter folder...")
    create_directory(anatomical_registration_dir, "Creating anatomical registration folder...") 
    create_directory(standard_registration_dir, "Creating standard registration folder...") 

    func_image = original_func_image(data_path, subject)
    
    # Drop first slices  
    drop_first_slices_image = drop_first_slices(drop_first_slices_dir, func_image)
    
    # Motion correction
    motion_correction_image = motion_correction(motion_correction_dir, drop_first_slices_image)

    # Slice timing correction
    slice_timming_correction_image = slice_timing_correction(slice_timing_correction_dir, motion_correction_image)
    
    # Brain extraction
    brain_extraction_image = brain_extraction(brain_extraction_dir, slice_timming_correction_image)

    # Intensity normalization
    intensity_normalization_image = intensity_normalization(intensity_normalization_dir, brain_extraction_image)

    # Nuisance signal regression
    subject_anat_dir = os.path.join(base_dir, subject, "anat")
    subject_anat_image = os.path.join(subject_anat_dir, 'segmentation_tissue', 'brain_extraction.nii.gz')
    clean_image = nuisance_reggresion(nuisance_reggresion_dir, intensity_normalization_image, motion_correction_dir, subject_anat_dir, subject_anat_image)
    
    # Band pass filter
    filter_image = band_pass_filter(band_pass_filter_dir, clean_image)

    # Registration to anatomical
    registration_anat_image = anatomical_registration(anatomical_registration_dir, filter_image, subject_anat_image)

    registration_standard_image = standard_registration(standard_registration_dir, subject_anat_dir, registration_anat_image, MNI_standard)

    rois = rois_extraction(subject, registration_standard_image, atlas_name)

# Main Pipeline Function
def main(base_dir, output_dir, data_path, subject, atlas_name):
    # Create working folder
    create_directory(base_dir, "Creating preprocessing/ folder...")
    
    # Create output folder
    create_directory(output_dir, "Creating output/ folder...")
    
    # Create output subfolders
    create_directory(os.path.join(output_dir, "rois_cc200"), "Creating output/rois_cc200/ folder...")
    create_directory(os.path.join(output_dir, "rois_aal"), "Creating output/rois_aal/ folder...")
    create_directory(os.path.join(output_dir, "rois_ho"), "Creating output/rois_ho/ folder...")
    
    # Create subject folder
    create_directory(os.path.join(base_dir, subject), f"Creating {subject}/ folder...")
   
    anat_prep(base_dir, data_path, subject)
    func_prep(base_dir, data_path, subject, atlas_name) 

# Example Usage
if __name__ == "__main__":
    base_dir = 'preprocessing'
    output_dir = 'output'
    data_path = 'data'
    subject = 'Caltech_51461'
    atlas_name = 'aal'

    main(base_dir, output_dir, data_path, subject, atlas_name)
