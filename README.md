# Deep-Learning-Challenge

## Overview - 
The goal of our Deep Learning Challenge is to help Alphabet Soup, a nonprofit foundation, to optimize its funding allocation process by developing a binary classifier using machine learning and neural network techniques. With an initial dataset containing records of over 34,000 organizations, the analysis involves preprocessing the data, partitioning it into training and testing subsets, and training a neural network model to predict the success likelihood of ventures funded by Alphabet Soup. Ultimately, the developed tool will empower Alphabet Soup to make informed decisions, directing funding towards ventures with the highest probability of success, thus maximizing its impact and achieving their philanthropic objectives.

## Results - 
* Data Preprocessing
  * Target Variable for the Model: The target variable selected for the model is the IS_SUCCESSFUL column, which indicates the success or failure of funding applications.
  * Features for the Model: The features chosen for the model include all columns from the dataset except for the IS_SUCCESSFUL column which include:  APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, and ASK_AMT. These features provide valuable information that the model will utilize to make predictions
  * Variables Removed from Input Data: Several variables were identified for removal from the input data as they are neither targets nor features. These variables include the EIN and NAME columns, which are identifiers with no predictive value. Additionally, preprocessing steps were applied to the APPLICATION_TYPE and CLASSIFICATION columns. For APPLICATION_TYPE, values with frequencies below 500 were grouped into an 'OTHER' category to reduce dimensionality and improve model performance. Similarly, for CLASSIFICATION, categories with frequencies under 1800 were aggregated into an 'other' category to enhance the model's predictive capabilities.

* Compiling, Training, and Evaluating the Original Model
 * How many neurons, layers, and activation functions did you select for your neural network model, and why?
 * Number of Neurons: The first hidden layer (layer_1) contains 80 neurons, and the second hidden layer (layer_2) contains 30 neurons. 
 * Number of Layers: The model consists of two hidden layers and one output layer, resulting in a total of three layers.
 * Activation Functions: Both hidden layers utilize the ReLU (Rectified Linear Unit) activation function, while the output layer employs the sigmoid activation function.
 * The selected architecture was aimed at striking a balance between model complexity and generalization performance, with the specific choices informed by experimentation and best practices in neural network design.

* Were you able to achieve the target model performance?
 * No, I was not able to achieve the target model performance of over 75% accuracy. Despite making several optimization attempts, the model's accuracy remained below the desired threshold. My best-performing model achieved an accuracy of approximately 72%, still short of the target. The optimization attempts yielded the following results: Optimization 1 - Loss: 0.574, Accuracy: 0.721. Optimization 2 - Loss: 0.572, Accuracy: 0.720. Optimization 3 - Loss: 0.582, Accuracy: 0.719. While the models showed improvement during optimization, they still did not reach the desired level of accuracy. Further analysis and experimentation may be required to identify additional strategies for enhancing model performance.

* What steps did you take in your attempts to increase model performance?
 * In attempts to enhance model performance, I made several modifications. Initially, I refined the dataset by removing additional columns ('USE_CASE', 'STATUS', 'SPECIAL_CONSIDERATIONS') and adjusted the cutoff values for 'application_type' and 'classification'. In Optimization 1, I increased the units in the first and second hidden layers and updated the training epochs to 50. For Optimization 2, I introduced a third hidden layer and switched the optimizer to 'sgd'. In Optimization 3, I added a fourth layer with updated units and changed the activation functions of the third and fourth layers to 'ELU'. Additionally, I reverted the optimizer to 'adam' and extended the training epochs to 100.

## Sumamry 
The deep learning model showed moderate performance in classifying the dataset, achieving an accuracy of approximately 72%. While this accuracy indicates some success, it falls short of the desired 75%. Despite multiple optimization attempts, including adjustments to model architecture, dataset preprocessing, and training parameters, the desired accuracy level was not attained. To address this problem more effectively, I recommend exploring a boosting algorithm, such as XGBoost or LightGBM. Boosting algorithms are known to help the performance in handling tabular data classification tasks. They capture complex relationships in the data, handling categorical variables effectively, and mitigating overfitting through ensemble techniques. By leveraging the strengths of a boosting algorithms, we can potentially achieve higher classification accuracy and enhance predictive performance in this scenario.

### Reference 
KDnuggets. (2022, July). Boosting Machine Learning Algorithms: An Overview. Retrieved from https://www.kdnuggets.com/2022/07/boosting-machine-learning-algorithms-overview.html
