# waste-segregation_model
A waste segregation model for 3.1 trisemester at ALU

**Project Overview**

This project aims to explore the implementation of Machine Learning Models with regularization, optimization, and Error analysis techniques used in machine learning to improve models' performance, convergence speed, and efficiency.

**Dataset Used**

I used a dataset from Kaggle, available at (https://www.kaggle.com/datasets/techsash/waste-classification-data). Due to the large size of the dataset, I utilized only the images from the **test** directory for this project. This approach allowed me to reduce computational load while still being able to evaluate the model effectively.

**Key Findings**
In this project, various optimization techniques were employed to improve the performance of the model, focusing on regularization, optimizers, early stopping, and dropout. These techniques are crucial in training deep learning models to achieve high accuracy while preventing overfitting. Here's a detailed discussion of each technique used, along with an explanation of the parameters, how they were tuned, and their significance in this context:

### 1. **Regularization Techniques (L1 and L2)**
Regularization is a technique used to prevent overfitting by penalizing large weights in the model. In this project, both L1 and L2 regularization methods were applied in different Configurations.

- **L1 Regularization (Lasso)**:
  L1 regularization adds the absolute value of the weights to the loss function. It encourages sparsity in the model by driving some weights to zero, effectively performing feature selection.
  - **Relevance**: In this project, L1 regularization helped control the complexity of the model by eliminating irrelevant or less important features, making the model simpler and reducing overfitting. The accuracy observed with L1 regularization (92.97% with Adam and 95.92% with RMSprop) shows its ability to improve model generalization while maintaining acceptable loss values.

- **L2 Regularization (Ridge)**:
  L2 regularization adds the square of the weights to the loss function. Unlike L1, it penalizes large weights but does not encourage sparsity. Instead, it tends to distribute the weights more evenly across all features.
  - **Relevance**: L2 regularization helped to smooth the learned weights, which reduces the impact of any individual feature dominating the model. The highest accuracy (96.17%) and the lowest loss (0.1032) were achieved with L2 regularization combined with Adam, indicating that this Configuration led to the best balance between generalization and performance.
  
  **Parameter Tuning**: Both regularizers have a regularization factor, commonly referred to as the **lambda (λ)**, which controls the strength of the regularization. A higher λ enforces stronger regularization, but overly large values might lead to underfitting. In this project, the λ value was carefully tuned through experimentation to strike a balance between reducing overfitting and maintaining model performance.

### 2. **Optimizers (Adam and RMSprop)**
Optimizers are algorithms used to update the model weights based on the gradients calculated during backpropagation. Two optimizers were used:

- **Adam (Adaptive Moment Estimation)**:
  Adam combines the benefits of two other optimizers, AdaGrad (which works well with sparse gradients) and RMSprop (which works well in non-stationary environments). Adam maintains a moving average of both the gradients and the squared gradients, which helps in adaptive learning rates.
  - **Relevance**: Adam was used in Configuration 1 and 4. It tends to converge faster and works well in scenarios with noisy gradients or when parameters are updated frequently. Configuration 4 (Adam + L2) resulted in the highest accuracy (96.17%) and the lowest loss (0.1032), demonstrating Adam's ability to effectively find the optimal parameters when combined with L2 regularization.

- **RMSprop (Root Mean Square Propagation)**:
  RMSprop is an adaptive learning rate optimizer that adjusts the learning rate based on the average of recent gradients, which helps in handling the exploding or vanishing gradient problem.
  - **Relevance**: RMSprop was used in Configuration 2 and 3. It performed particularly well with L1 regularization, achieving 95.92% accuracy with a low loss of 0.1391. This shows that RMSprop, when paired with L1 regularization, was effective in preventing overfitting while maintaining high accuracy.

  **Parameter Tuning**: Both optimizers have **learning rate** as a key parameter. The learning rate controls how much the weights are adjusted during each update. A higher learning rate speeds up convergence but can overshoot the optimal point, while a lower learning rate may lead to slow convergence or getting stuck in local minima. In this project, the learning rate was tuned through trial and error, and the default values (Adam: 0.001, RMSprop: 0.001) worked well after experimentation.

### 3. **Early Stopping**
Early stopping is a regularization technique that stops training when the model's performance on a validation set stops improving. This helps in preventing overfitting since the model is stopped before it starts to learn noise from the training data.

- **Relevance**: Early stopping was applied in all Configurations and played a significant role in optimizing the model's performance by stopping training at the optimal point. For instance, Configurations 2 and 4 achieved high accuracy with relatively low loss, and early stopping ensured that these models did not overfit, even when trained for fewer epochs.

  **Parameter Tuning**: The **patience** parameter was tuned to determine how many epochs the model would wait for an improvement before stopping. A patience value of 3-5 was selected after experimentation, which allowed the model to converge to the best performance without prematurely halting training.

### 4. **Dropout**
Dropout is another regularization technique that prevents overfitting by randomly setting a fraction of input units to zero at each update during training. This forces the model to learn more robust features by not relying too heavily on any one node.

- **Relevance**: Dropout was applied in all Configurations to improve the generalization of the model. By randomly dropping units during training, dropout helped to prevent overfitting, which was particularly beneficial when combined with early stopping and regularization. For instance, the Configuration of L2 regularization with dropout in Configuration 4 led to the best overall performance.

  **Parameter Tuning**: The **dropout rate** was tuned to find the optimal level of regularization. A dropout rate of 0.2 to 0.5 is commonly used in practice, and a value of around 0.3 was found to work best in this project, offering a good tradeoff between regularization and model capacity.

### Summary of Results and Conclusion
- **Configuration 1 (L1 + Adam + Early Stopping + Dropout)**: While achieving 92.97% accuracy and a loss of 0.2125, this Configuration had decent performance, but Adam struggled slightly with L1 regularization compared to RMSprop.
  
- **Configuration 2 (L1 + RMSprop + Early Stopping + Dropout)**: This Configuration performed well with 95.92% accuracy and a loss of 0.1391. RMSprop paired better with L1 regularization, leading to better generalization and lower loss.

- **Configuration 3 (L2 + RMSprop + Early Stopping + Dropout)**: L2 regularization with RMSprop yielded good results, with 94.30% accuracy and a loss of 0.1424, but not as strong as Configuration 4.

- **Configuration 4 (L2 + Adam + Early Stopping + Dropout)**: This Configuration resulted in the best performance, with 96.17% accuracy and the lowest loss of 0.1032. The synergy between Adam and L2 regularization, along with early stopping and dropout, led to the most effective model.

Thus, the Configuration of **L2 regularization, Adam optimizer, early stopping, and dropout** was the best optimization strategy for this project. The parameter choices, including the learning rate, regularization strength, dropout rate, and early stopping patience, were tuned carefully through experimentation, leading to this optimal result.

**ERROR ANALYSIS**

