import streamlit as st

# URL of the logo
logo_url = "https://www.fedecardio.org/wp-content/uploads/2021/03/schema-valves_0.jpg"

# Display the logo
st.image(logo_url, width=100)

# Title of the app
st.title('Simple Summation App')

# Input fields for x and y
x = st.number_input('Enter the value of x:', value=0)
y = st.number_input('Enter the value of y:', value=0)

# Calculate the sum
sum_result = x + y

# Display the result
st.write(f'The sum of {x} and {y} is {sum_result}')
