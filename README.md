## Meta Modeller for Supervized Machine Learning
### Abstract
Data Science/Machine Learning is a field which is changing at the fast pace with many
research is being conducted in this area. The big players like Amazon and Microsoft are
offering tool sets to carry out Model building and Deployment in the cloud at the click of a
button. The model building is fairly a linear approach but, it takes many iterations to arrive at
the best fitting hypothesis for the business use case. All the iterations are called experiments
(hypothesis testing), they consume both time and money before arriving at a best suitable
solution.
The majority of the time is consumed in cleaning the data, pre-processing the data and
feature engineering (selection, extraction). Once the data is ready to be consumed by a
machine learning model, the Data Scientist/Machine Learning Engineer need to test multiple
hypothesis (models) to choose the best model. The complexity of this process increases
with the number of hyper-parameters a particular model has and the number of models we
are testing. Though this process is unavoidable, it can be streamlined in a way so it offloads
the repetitive defined tasks on the Data Scientist by doing them in an automated way.
This project delivers a framework where the Data Scientist/Machine Learning Engineer can
use it for end to end model building, in this case for supervised models (Regression and
Classification). The framework is a pipeline which automates Data Preprocessing and Model
Building combined with Hyper-parameter tuning. The planned output of this framework is a
best performing model with the attained metric along with the learned hyper-parameters.
This scope of this framework is supervised learning models Regression and Classification.
To demonstrate the ability of this project four models for regression and four models for
Classification will be trained. The model mix contains both distance based and tree based
models. As part of Data Preprocessing, the model agnostic tasks like handling missing
values, handling skew, handling outliers, feature scaling will be implemented. During model
building, the tasks feature selection, model selection, hyper-parameter tuning and cross
validation will be implemented.
Currently the organisations are using licensed or subscription based tools in order to
improve the productivity for a similar kind of tools. Building this framework in python helps
the organisation to reduce licensing costs and gives the flexibility to modify the tool as per
their requirements. Additionally, this framework can be enhanced to include any new
algorithms or libraries when available in the market. It gives the organisation total control
over the tool besides saving the cost.
