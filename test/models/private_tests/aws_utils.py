import pytest
import boto3
import os
import shutil
import tempfile

BUCKET_NAME = 'tensorleap-engine-tests-dev'
PREFIX = 'onnx2keras'
s3 = boto3.client(
    's3',
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    aws_session_token=os.environ['AWS_SESSION_TOKEN'],
    region_name='us-east-1'
)

@pytest.fixture
def aws_s3_download(request):
    def download_from_s3(aws_dir, dest_dir="", is_temp=False):
        real_dir = ""
        if not is_temp:
            real_dir = dest_dir
            if len(dest_dir) == 0:
                raise Exception("Need to provide destination dir if non-temp directory is used for file downloading")
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir, exist_ok=True)
        else:
            # Create a temporary directory
            real_dir = tempfile.mkdtemp()
        path = f"{PREFIX}/{aws_dir}"  # Use the provided directory as the prefix
        # List objects in the bucket with the specified prefix
        response = s3.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=path
        )

        # Download files to the temporary directory
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                if obj['Size'] > 0:
                    rel_path = key[len(path):].lstrip("/")
                    dirname = os.path.dirname(rel_path)
                    full_dir = os.path.join(real_dir, dirname)
                    if len(full_dir) > 0 and not os.path.exists(full_dir):
                        os.makedirs(full_dir)
                    filename = os.path.join(real_dir, rel_path)
                    if not os.path.exists(filename):
                        s3.download_file(BUCKET_NAME, key, filename)
                        print(f"Downloaded {key} to {filename}")
        else:
            print("No objects found under the specified prefix.")

        # Provide the temporary directory path to the test function
        return real_dir, is_temp

    # Yield the download_from_s3 function so it can be used as a fixture
    dir_path, is_temp = download_from_s3(*request.param)
    yield dir_path

    # Clean up the temporary directory after the test
    if is_temp:
        shutil.rmtree(dir_path)
