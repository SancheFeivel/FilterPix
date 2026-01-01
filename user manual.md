### \# FILTERPIX USER MANUAL

### 

### \## OVERVIEW

### FilterPix automatically filters large photo collections by removing blurry images, organizing burst sequences, and sorting images by content using AI. The application \*\*copies\*\* selected files into organized output folders while leaving original files untouched.

### 

### ---

### 

### \## MAIN SCREEN

### 

### \### Folder Selection

### \- \*\*Input Button\*\*: Select the folder containing images to be processed  

### \- \*\*Folder Button\*\*: Opens the selected folder in the system file explorer  

### 

### \### Processing Controls

### \- \*\*Start\*\*: Begins filtering using the current configuration  

### \- \*\*Cancel\*\*: Stops processing at any time  

### 

### \### Process Report

### Displayed after processing completes:

### \- Number of images processed, selected, and rejected  

### \- Total time elapsed  

### 

### ---

### 

### \## SETTINGS

### 

### \### Output Settings

### 

### \#### Destination Folder

### \- \*\*Default\*\*: Automatically creates `Sharp` and `Sorted` folders inside the selected input directory  

### \- \*\*Custom\*\*: Allows selection of a different output location  

### 

### ---

### 

### \## FILTER OPTIONS

### 

### \### Rating Filter

### \- Checks image metadata for star ratings  

### \- Keeps only images that have a rating  

### \- Useful if images were previously rated in-camera or using external software  

### 

### \### Sharpness Filter

### \- Uses a Laplacian-based edge detection algorithm to identify and remove blurry images  

### 

### \### Burst Selection

### \- Detects burst sequences and selects the best two images per sequence  

### \- Intended for action photography and rapid shooting scenarios  

### 

### \### Sharpness Threshold

### \- \*\*Low\*\*: Lenient, retains more images  

### \- \*\*Med\*\*: Balanced (default)  

### \- \*\*High\*\*: Strict, keeps only the sharpest images  

### 

### \### Threshold Compensation

### \- Fine-tunes sharpness filtering with a custom integer value  

### \- Positive values increase strictness (e.g., `+20`)  

### \- Negative values reduce strictness (e.g., `-20`)  

### \- Empty or invalid values default to `0`  

### 

### ---

### 

### \## SORT SETTINGS

### 

### \### Image Detection

### \- AI-based image sorting using content recognition  

### \- Increases processing time and system resource usage  

### 

### \### Detection Mode

### \- \*\*Fast\*\*: Faster processing with reduced accuracy  

### \- \*\*Accurate\*\*: Slower processing with improved recognition accuracy  

### 

### ---

### 

### \## RECOMMENDED WORKFLOWS

### 

### \### Standard Workflow (Recommended)

### \- Rating Filter  

### \- Sharpness Filter  

### \- Burst Selection  

### \- Threshold: \*\*Med\*\*  

### \- Image Detection: \*\*Disabled\*\*  

### 

### \### Quick Cleanup

### \- Sharpness Filter only  

### \- Threshold: \*\*High\*\*  

### \- Optimized for speed  

### 

### \### Maximum Quality Filtering

### \- All filters enabled  

### \- Image Detection set to \*\*Accurate\*\*  

### \- Highest accuracy, slowest processing  

### 

### \### Burst Photography

### \- Burst Selection  

### \- Sharpness Filter set to \*\*Med\*\* or \*\*High\*\*  

### \- Optimized for action and sports photography  

### 

### ---

### 

### \## OUTPUT STRUCTURE

### \- \*\*Sharp\*\*: Images that pass all enabled filters  

### \- \*\*Sorted\*\*: Rejected images (blurry, duplicates, or below threshold)  

### 

### The original image folder is \*\*never modified\*\*. FilterPix only creates copied outputs.

### 

### ---

### 

### \## TROUBLESHOOTING

### 

### \### Excessive Image Rejection

### \- Lower sharpness threshold to \*\*Low\*\*  

### \- Apply negative threshold compensation (e.g., `-20`)  

### 

### \### Insufficient Image Rejection

### \- Increase threshold to \*\*High\*\*  

### \- Apply positive threshold compensation (e.g., `+20`)  

### 

### \### Slow Performance

### \- Disable Image Detection  

### \- Use \*\*Fast\*\* detection mode if detection is required  

### 

### ---

### 

### \## TECHNICAL NOTES

### \- Supports JPG, JPEG, PNG, and other common image formats  

### \- Original image files are never altered or deleted  

### \- Ensure sufficient disk space is available for copied output files  

### 

