#!/bin/bash

cd EnsembleLearning
printf "Ensemble Learning Problems\n"
printf "Problem 2a\n"
python3 cs6350_hw2_p2a.py
printf "Problem 2b\n"
python3 cs6350_hw2_p2b.py
printf "Problem 2c\n"
python3 cs6350_hw2_p2c.py
printf "Problem 2d\n"
python3 cs6350_hw2_p2d.py
printf "Problem 2e\n"
python3 cs6350_hw2_p2e.py

cd ../LinearRegression
printf "Linear Regression Problems\n"
printf "Problem 4a\n"
python3 linear_regression.py
printf "Problem 4b\n"
python3 stochastic_grad_descent.py
printf "Problem 4c\n"
python3 analytical_regression.py

