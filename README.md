# Bixi-Montreal-data-analysis
analysis on various variables' effect on number of bike-rent trips 
## A. Procedures
• load bike rental data 'data_2019' from year 2019 and conduct data manipulation and visualization to find the most and least used bus station in Montreal <br> 
• load and merge 'Weather' data to retrive the relationship between various effet on total number of trips within one-hour-interval <br>
• Implemented statistical procedures: Linear Regression model, K-nearest neighbors method and Ridge Regression model to estimate and to forecast responses from multivariable factors <br>



## B. Conclusions
• The most frequently used bus station in Montreal are: Métro Laurier (Rivard / Laurier); Métro Mont-Royal (Rivard / du Mont-Royal); de la Commune / King <br>
• The least frequently used bus station in Montreal are: St-Charles / Grant; St-Charles / St-Jean; 8e avenue / Notre-Dame Lachine <br>
• people tend to rent bikes more on weekdays than weekend; people tend to rent bikes more on rush hours  <br>
• people tend to go out on cloudy and clear days; people tend not to go out on drizzle and foggy days <br>
• the most important factor influencing num_trips among all nine factors is temperature <br>
• the k-nearest neighbor testing tells that k=11 will give us the best result in this case <br>
• the clf ridge prediction model fits the data quite well with 65% (close to 1) fitness and 0.014 (close to 0)mean squared error <br>
