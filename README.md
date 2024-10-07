
# Project: Collaborative Filtering Algorithm Implementation and Evaluation

### Objective
This project implements and evaluates collaborative filtering techniques for a movie recommendation system using 5-star user rating data. The primary goal is to develop robust collaborative filtering algorithms and enhance predictive accuracy through a hybrid post-processing approach.

### Methodology
The project implements two main collaborative filtering methods from scratch, followed by a post-processing step using SVD with Kernel Ridge Regression (KRR) to refine predictions.

1. **Collaborative Filtering Algorithms**:
   - **Gradient Descent with Probabilistic Assumptions**: This approach models user-item interactions by assuming a probabilistic distribution for user preferences. The algorithm minimizes a regularized loss function:
     \[
     L = \sum_{(u,i) \in \text{observed}} (r_{ui} - \hat{r}_{ui})^2 + \lambda (\|\mathbf{P}\|^2 + \|\mathbf{Q}\|^2)
     \]
     where \( r_{ui} \) is the actual rating, \( \hat{r}_{ui} = \mathbf{p}_u^T \mathbf{q}_i \) is the predicted rating for user \( u \) and item \( i \), and \( \lambda \) is the regularization term. Here, \(\mathbf{P}\) and \(\mathbf{Q}\) represent user and item latent factor matrices, respectively. Gradient descent iteratively updates these matrices to minimize prediction error. The learning rate, regularization strength, and the number of latent dimensions were tuned using grid search and cross-validation.

   - **Alternating Least Squares (ALS)**: This approach alternates between optimizing user and item latent factors by solving for each in a closed-form while holding the other fixed. In each iteration, the following updates are performed:
     \[
     \mathbf{p}_u = \left( \sum_{i \in I_u} \mathbf{q}_i \mathbf{q}_i^T + \lambda \mathbf{I} \right)^{-1} \sum_{i \in I_u} r_{ui} \mathbf{q}_i
     \]
     \[
     \mathbf{q}_i = \left( \sum_{u \in U_i} \mathbf{p}_u \mathbf{p}_u^T + \lambda \mathbf{I} \right)^{-1} \sum_{u \in U_i} r_{ui} \mathbf{p}_u
     \]
     Here, \(I_u\) is the set of items rated by user \(u\), \(U_i\) is the set of users who rated item \(i\), and \(\mathbf{I}\) is the identity matrix used for regularization. ALS enables efficient factor updates by exploiting the least squares solution, making it computationally advantageous for large-scale data.

2. **Post-Processing with SVD and Kernel Ridge Regression (KRR)**:
   - **Singular Value Decomposition (SVD)**: SVD was applied to the user-item interaction matrix to decompose it into lower-rank matrices. This decomposition captures latent structures in the data and reduces noise. The decomposed matrix can be expressed as:
     \[
     \mathbf{R} \approx \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T
     \]
     where \(\mathbf{U}\) and \(\mathbf{V}\) are orthogonal matrices representing users and items in the reduced space, and \(\mathbf{\Sigma}\) is a diagonal matrix of singular values.
   
   - **Kernel Ridge Regression (KRR)**: KRR was used to enhance the predictions by modeling non-linear relationships within the SVD-transformed space. The KRR loss function combines the ridge penalty with kernel methods:
     \[
     \min_{\mathbf{\alpha}} \| \mathbf{y} - K \mathbf{\alpha} \|^2 + \lambda \| \mathbf{\alpha} \|^2
     \]
     where \( K \) is the kernel matrix, and \( \lambda \) is the regularization term. We used a Gaussian (RBF) kernel to capture complex patterns among latent features obtained from SVD. This post-processing step was applied consistently across both collaborative filtering methods to improve the precision of final predictions.

3. **Model Evaluation and Comparison**:
   - **Evaluation Metrics**: Each model’s performance was evaluated using Root Mean Square Error (RMSE) on the test set, as well as Mean Absolute Error (MAE) for an alternative view on prediction accuracy. These metrics were computed as follows:
     \[
     \text{RMSE} = \sqrt{\frac{1}{N} \sum_{(u,i) \in \text{test}} (r_{ui} - \hat{r}_{ui})^2}
     \]
     \[
     \text{MAE} = \frac{1}{N} \sum_{(u,i) \in \text{test}} |r_{ui} - \hat{r}_{ui}|
     \]
     where \(N\) is the number of test ratings. Both metrics provide insight into the model's ability to generalize to unseen user-item interactions.

   - **Model Comparison**: The performance of ALS and gradient descent-based collaborative filtering was assessed both with and without the SVD+KRR post-processing. We observed the impact of each method on predictive accuracy and computational cost. ALS, given its closed-form solution, was faster and more scalable, while gradient descent offered a more flexible optimization framework.

### Technical Implementation
- **Data Handling and Preprocessing**: The user-item matrix was preprocessed to account for sparsity. Missing ratings were initialized with zero, and matrix completion was handled through each collaborative filtering algorithm. Data standardization was applied to maintain consistency across models.
- **Hyperparameter Tuning**: We applied grid search and k-fold cross-validation to identify optimal hyperparameters, including learning rate, regularization strength, and the number of latent factors.
- **Code Structure**: The codebase was modular, with separate scripts for data loading, model training, and evaluation. This modularity allows easy experimentation with different model parameters and configurations.

### Project Structure
The repository is organized to facilitate reproducibility and modular testing:

```
proj/
├── lib/        # Python scripts for collaborative filtering, SVD, and KRR post-processing
├── data/       # Raw and processed datasets, including scripts for data preprocessing
├── doc/        # Detailed documentation, reports, and methodological descriptions
├── figs/       # Visualizations of model performance metrics, comparison plots, and error distributions
└── output/     # Model predictions, performance summaries, and serialized models for deployment
```

### Results and Future Directions
The implementation demonstrated the effectiveness of ALS and gradient descent for collaborative filtering, with the SVD+KRR post-processing step enhancing predictive accuracy. The ALS method, combined with SVD+KRR, yielded the lowest RMSE, showing the potential for scaling and high performance in real-world recommendation systems. Future directions include:
- **Deep Learning Approaches**: Integrating neural network-based collaborative filtering, such as neural matrix factorization, could capture more complex user-item interactions.
- **Incorporation of Additional Metadata**: Adding side information, such as user demographics or item genres, could further personalize recommendations.
- **Scalability Optimization**: Optimizing matrix factorization implementations for large datasets and exploring distributed processing approaches to handle extremely large-scale recommendation tasks.

