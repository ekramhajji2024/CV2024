import streamlit as st

# Title of the app
st.title('Simple Summation App')

# Input fields for x and y
x = st.number_input('Enter the value of x:', value=0)
y = st.number_input('Enter the value of y:', value=0)

# Calculate the sum
sum_result = x + y

# Display the result
st.write(f'The sum of {x} and {y} is {sum_result}')
