
default: pylint

pylint:
	find . -iname "*.py" -not -path "./tests/*" | xargs -n1 -I {}  pylint --output-format=colorized {}; true
# ----------------------------------
#         LOCAL SET UP
# ----------------------------------

run_locally:
	@python -W ignore -m TaxiFareModel.trainer

install_requirements:
	@pip install -r requirements.txt

# ----------------------------------
#    LOCAL INSTALL COMMANDS
# ----------------------------------
install:
	@pip install . -U


clean:
	@rm -fr */__pycache__
	@rm -fr __init__.py
	@rm -fr build
	@rm -fr dist
	@rm -fr TaxiFareModel-*.dist-info
	@rm -fr TaxiFareModel.egg-info
	-@rm model.joblib

#-------------------------------------
#               GCP
#-------------------------------------
##### Machine configuration - - - - - - - - - - - - - - - -

PYTHON_VERSION=3.7
FRAMEWORK=scikit-learn
RUNTIME_VERSION=1.15

##### Package params  - - - - - - - - - - - - - - - - - - -

PACKAGE_NAME=SimpleTaxiFare
FILENAME=trainer

# project id - replace with your GCP project id
PROJECT_ID='teak-store-332410'

# bucket name - replace with your GCP bucket name
BUCKET_NAME='wagon-data-722-ferrette'

# choose your region from https://cloud.google.com/storage/docs/locations#available_locations
REGION=europe-west1

set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}


# path to the file to upload to GCP (the path to the file should be absolute or should match the directory where the make command is ran)
LOCAL_PATH='/Users/corinneferrette/code/corinneferrette/TaxiFareModel/raw_data/train_1k.csv'

# bucket directory in which to store the uploaded file (`data` is an arbitrary name that we choose to use)
BUCKET_FOLDER=data

# name for the uploaded file inside of the bucket (we choose not to rename the file that we upload)
BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})
# will store the packages uploaded to GCP for the training
BUCKET_TRAINING_FOLDER = 'trainings'
PACKAGE_NAME=TaxiFareModel
upload_data:
    # @gsutil cp train_1k.csv gs://wagon-ml-my-bucket-name/data/train_1k.csv
	@gsutil cp ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}

save_model:
	@gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER}  \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs

# - - - - - - - - - - - - - - - - - - - - - - - - -
#
# Cloud Functions

FUNCTION_NAME=hacker-scrap
REGION=europe-west1
CODE_SOURCE_PATH=.
CODE_ENTRY_POINT=storage_upload
RUNTIME=python39
TIMEOUT=60s

deploy_function:
	gcloud functions deploy ${FUNCTION_NAME} \
		--region ${REGION} \
		--trigger-http \
		--no-allow-unauthenticated \
		--source ${CODE_SOURCE_PATH} \
		--entry-point ${CODE_ENTRY_POINT} \
		--runtime ${RUNTIME} \
		--timeout ${TIMEOUT}

list_function:
	gcloud functions list

describe_function:
	gcloud functions describe ${FUNCTION_NAME} \
		--region ${REGION} \

# - - - - - - - - - - - - - - - - - - - - - - - - -
#
# Cloud Scheduler

JOB_NAME=job-name
JOB_FREQUENCY="* * * * *"
FUNCTION_URI="https://europe-west1-le-wagon-data.cloudfunctions.net/hacker-scrap"
SERVICE_ACCOUNT_EMAIL=le-wagon-data@le-wagon-data.iam.gserviceaccount.com

deploy_trigger:
	gcloud scheduler jobs create http ${JOB_NAME} \
		--schedule ${JOB_FREQUENCY} \
		--uri ${FUNCTION_URI} \
		--oidc-service-account-email ${SERVICE_ACCOUNT_EMAIL}

pause_trigger:
	gcloud scheduler jobs pause ${JOB_NAME}

resume_trigger:
	gcloud scheduler jobs resume ${JOB_NAME}

delete_trigger:
	gcloud scheduler jobs delete ${JOB_NAME} --quiet

list_trigger:
	gcloud scheduler jobs list

describe_trigger:
	gcloud scheduler jobs describe ${JOB_NAME}
