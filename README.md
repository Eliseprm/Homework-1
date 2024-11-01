# Homework-1

# Problem 1 (10 Pts)

This question tests your ability to follow instructions. If you follow the instructions for this assignment, you will receive full credit for this question.

a. List the names of your group [2 pts]

b. Type your answers (meaning you did not submit your assignment written by hand) [2 pts]

c. Format your assignment in the form of a write-up, not simply code that you submit (including proper formatting and labelling of tables and graphs) [2 pts]

d. Provide the supporting Python code for your answers [2 pt]

e. Your Python code actually works on its own and provides the answers that you submitted in your
write-up [2 pts]

# Problem 2 - Value-at-Risk (35Pts)

For this question, we will be looking at some real-world data and calculating some risk measures. To begin, get return data for the value-weighted market return with dividend distributions along with stocks of your choice that each start with the first letter of your team members names. For example, if you had a team whose members included Emily, John, and Jack you would need to get data for three stocks whose tickers start with E, J, and J. Your data should be daily and range from December 2nd 2013 to December 2nd 2022. Your best source for this data will be WRDS, nut you will need access, which can take a couple of days. Answer the following questions:

a. Provide descriptive statistics and time-series graphs for your data. Make sure to label your graphs correctly and discuss the results (both the descriptive statistics and the graphs).[5 pts]

b. Write a function that consists of two inputs: 1) a 1-D array-like object of returns data 2) a real value α that is bounded by zero and one and that will be used to estimate the V aRα and the ESα of the 1-D array. Get some results from your function and provide their interpretations. [5 pts]

c. Write a function that will calculate the VaR for the OLS fitted values assuming iid normal errors: ε ∼ i.i.d. N(0,σOLS). Make sure that your function accepts an object of type RegressionResults and a real value p that is bounded by zero and one. Get some results from your function and provide their interpretations. [5 pts]

d. Calculate the VaR just like you did in part (c), but this time, we will make the assumption that the σ÷2 σ2

regression errors are defined as: σ2 = OLS + t−1 and σ2 = σ’2 . Interpret what you find. What type of model is this? How are the results different from what you found in part (c)? [10 pts]

e. Write a function that will allow you to test the null hypothesis that the value-at-risk has coverage rate p (note that p should be bounded by zero and one) for the return array. You can use the formula for s1 in the slides for lecture 2 for the test statistic and the exact Binomial distribution for the p-value. Interpret your results.[5 pts]

f. Write a function that calculates the Wald-Wolfowitz statistic as well as its p-value. To calculate the p-value, use the large sample approximation; see example 15-14 in Mann’s chapter. What do you conclude based on what you found? Interpret your result.[5 pts]

# Problem 3 - Simulations

For this question, in order to get credit, you will need to 1) write the functions and 2) interpret the results obtained from the functions you wrote. Make sure to provide context for your interpretation of the result: meaning just writing “I got an answer of 1.21029” will not get you any credit for your interpretation. You need to tell me what the result means in the context of what we discuss in this course.

a. Write a function that accepts [10 pts]:

• an object of type BitGenerator, BG

• a positive integer degrees and returns

• a 1-D array of 200 i.i.d. draws from a studentt distribution with degrees degrees of freedom. 

Do not seed the BitGenerator in your function.

b. Write a function that accepts [5 pts]:
• a BitGenerator bg • a 1-D array, a
• a boolean, r
and returns
• a 1-D array of the same size as a that contains values randomly drawn from a
• those values are randomly drawn with replacement if replace = True and otherwise are drawn without replacement (i.e. they are “shuffled.”)

c. Write a function that accepts a 1-D array s and produces a probability plot comparing its values to a N(0,1) distribution. [5 pts]

d. Write a function that accepts [5 pts]:

• a BitGenerator bg

• a 1-D array of real values a • a positive integer T

and then

• bootstraps (samples with replacement) from a

• calculates the Kolmogorov-Smirnov test statistic for the H0 that the bootstrapped values follow a N(0,1) distribution

• repeats the above two steps T times. 

The function then returns
• p a real value scalar 0 ≤ p ≤ 1, the fraction of the T simulations where we reject H0 at the 5% significance level.

SUGGESTION: You could use Q2 Sample() to do the bootstrapping 

e. Explain briefly for each of the following [10 pts]:

• What do your results from part (d) tell you about the size and power of the KS test? 

• Are your results from part (d) consistent with your results from part (c)?

# Problem 4 - Combining Simulations and Value-at-Risk

Here we will combine what we did in questions 2 and 3.

a. Write a function that will calculate a bootstrapped VaR using 10,000 draws. Interpret the resulting VaR value. [5 pts]

b. Graph the results in part (a) and discuss what you find. [5 pts]

c. What is the 95% confidence interval for your VaR? [5 pts]

d. Compare the results of what you found in this question to the results of what you found in question 2. Are they the same? Why or why not. [5 pts]
























