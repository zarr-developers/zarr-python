$PROJECT = 'zarr'

$ACTIVITIES = ['version_bump']

$VERSION_BUMP_PATTERNS = [  # These note where/how to find the version numbers
                         ('zarr/version.py', r'__version__\s*=.*', "__version__ = '$VERSION'"),
                         ('setup.py', r'version\s*=.*,', "version='$VERSION',")
                         ]



$GITHUB_ORG = 'zarr'  # Github org for Github releases and conda-forge
$GITHUB_REPO = 'zarr-python'  # Github repo for Github releases  and conda-forge