# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

The resnet50 has been chosen as the model for this project because it has been pretrained using morethan 1 million of images from ImageNet database which contains good number of kernels to differenciate different types of images.

Three hyperparameters have been tested using hyperparameter tuning jobs
 - learning rate - the fastest rate that model will train without the overfitting
 - batch size - Time batch-size used for optimal level of processing time with the accuarcy
 - epochs - The optimal number of epochs to produce better crossentropy loss.

- Screenshots of completed training jobs
 - Hyperparameter Training job:
![image](https://user-images.githubusercontent.com/98076289/225024483-3238c4ea-3073-4158-87ab-a3bc371d136c.png)
 - Training Job With Optimal Hyperparameter Values
![image](https://user-images.githubusercontent.com/98076289/225025499-6794b3e6-c9f8-4cf8-8502-9f56e80f8035.png)

- Logs metrics during the training process
- Logs in Training Processes
![image](https://user-images.githubusercontent.com/98076289/225026512-4a3eb7be-26b4-4e85-9036-ed63920eebab.png)
- Instance Metrics
![image](https://user-images.githubusercontent.com/98076289/225027617-8fffbf33-b850-44d3-9165-1367742d76d6.png)

- learning rate, batch size and epochs hyperparameters have been tuned as  
"lr": ContinuousParameter(0.001, 0.1)  
"batch-size": CategoricalParameter([32, 64, 128, 256, 512])  
"epochs": IntegerParameter(1, 100)  

- Retrieve the best best hyperparameters from all your training jobs
![image](https://user-images.githubusercontent.com/98076289/225028831-f6e04025-4a54-41f4-a712-7283802628ab.png)


## Debugging and Profiling
Using train_model.py script a final model has been defined with debugging options. This training job use the best training model hyperparameters in hyperparameter tuning job.  
'batch-size': 64  
'epochs': 1  
 'lr': 0.001126550316150208    

### Results
What are the results/insights did you get by profiling/debugging your model?
Profiling report provides summary of training job, system resource usage statistics, framework metrics, rules summary, and detailed analysis from each rule.  
The profiler html/pdf file is[here](https://github.com/pubuduAeturnum/Udacity-ML-Scholarship-Project3/blob/main/ProfilerReport/profiler-output/profiler-report.html)  
The crossentropy loss has been decreased as below when the model has been training from the dogbreed image dataset.
![image](https://user-images.githubusercontent.com/98076289/225042403-a8d52534-0d9c-410d-a18b-f43e8d383540.png)



## Model Deployment
* To query the endpoint with sample input an entrypoint should be created to send the preprocessed image to it for this perpose **inference.py** python script has been used. Also the endpoint should be created using GPU supported instance type (ml.g4dn.4xlarge)
![image](https://user-images.githubusercontent.com/98076289/225051149-c8eabab2-ce9d-43b0-9e84-899eb0b86485.png)



## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
