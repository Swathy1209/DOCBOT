import pandas as pd
import streamlit as st
import plotly.express as px

# Load the health monitoring data
health_data = pd.read_csv("C:/Users/swathiga/Downloads/healthmonitoring (1).csv")
health_data['Timestamp'] = pd.to_datetime(health_data['Timestamp'])

# Load the doctors dataset and clean it
doctors_df = pd.read_csv("C:/Users/swathiga/Downloads/doctors_in_coimbatore.csv")
doctors_df = doctors_df.drop_duplicates()

# Set up the Streamlit app
st.title("DocBot - Proactive Health Management System")

# Ask for PatientID
patient_id = st.text_input("Enter Patient ID")

if patient_id:
    # Filter data for the given PatientID
    patient_data = health_data[health_data['PatientID'] == int(patient_id)]
    
    if patient_data.empty:
        st.error("Patient ID not found. Please enter a valid Patient ID.")
    else:
        # Display patient summary
        st.subheader("Patient Summary")
        st.write(f"Age: {patient_data['Age'].values[0]}")
        st.write(f"Gender: {patient_data['Gender'].values[0]}")
        
        # Calculate and display summary statistics
        st.subheader("Summary Statistics")
        st.write(patient_data.describe())
        
        # Handle HeartRate data
        health_data['HeartRate'] = pd.to_numeric(health_data['HeartRate'], errors='coerce')
        if health_data['HeartRate'].isnull().any():
            st.warning("Heart Rate data contains missing values. These will be excluded from the plot.")
            health_data = health_data.dropna(subset=['HeartRate'])

        # Plot Heart Rate over time
        st.subheader("Heart Rate Over Time")
        fig_heart_rate = px.line(patient_data, x='Timestamp', y='HeartRate', title='Heart Rate Over Time')
        st.plotly_chart(fig_heart_rate, use_container_width=True)
        
        # Split the BloodPressure into Systolic and Diastolic only for the relevant patient data
        if 'BloodPressure' in patient_data.columns:
            bp_split = patient_data['BloodPressure'].str.split('/', expand=True)
            patient_data['Systolic'] = pd.to_numeric(bp_split[0], errors='coerce')
            patient_data['Diastolic'] = pd.to_numeric(bp_split[1], errors='coerce')

            # Plot Systolic Blood Pressure Distribution
            if 'Systolic' in patient_data.columns and 'Diastolic' in patient_data.columns:
                st.subheader("Systolic Blood Pressure Distribution")
                fig_blood_pressure_dist = px.histogram(patient_data, x='Systolic', title='Systolic Blood Pressure Distribution')
                st.plotly_chart(fig_blood_pressure_dist, use_container_width=True)

                # Plot Diastolic Blood Pressure Distribution
                st.subheader("Diastolic Blood Pressure Distribution")
                fig_diastolic_pressure_dist = px.histogram(patient_data, x='Diastolic', title='Diastolic Blood Pressure Distribution')
                st.plotly_chart(fig_diastolic_pressure_dist, use_container_width=True)
            else:
                st.warning("Blood Pressure data is not available for the selected patient.")

        # Plot Activity Level Distribution
        st.subheader("Activity Level Distribution")
        fig_activity_level_dist = px.pie(health_data, names='ActivityLevel', title='Activity Level Distribution')
        st.plotly_chart(fig_activity_level_dist, use_container_width=True)

        # Plot Sleep Quality Distribution
        st.subheader("Sleep Quality Distribution")
        fig_sleep_quality_dist = px.pie(health_data, names='SleepQuality', title='Sleep Quality Distribution')
        st.plotly_chart(fig_sleep_quality_dist, use_container_width=True)

        # Plot Stress Level Distribution
        st.subheader("Stress Level Distribution")
        fig_stress_level_dist = px.pie(health_data, names='StressLevel', title='Stress Level Distribution')
        st.plotly_chart(fig_stress_level_dist, use_container_width=True)

        # Display a summary based on the latest entry
        latest_entry = patient_data.iloc[-1]
        summary = (
            f"Latest Recorded Data:\n"
            f"Heart Rate: {latest_entry['HeartRate']}\n"
            f"Blood Pressure: {latest_entry['BloodPressure']}\n"
            f"Respiratory Rate: {latest_entry['RespiratoryRate']}\n"
            f"Body Temperature: {latest_entry['BodyTemperature']}\n"
            f"Activity Level: {latest_entry['ActivityLevel']}\n"
            f"Oxygen Saturation: {latest_entry['OxygenSaturation']}\n"
            f"Sleep Quality: {latest_entry['SleepQuality']}\n"
            f"Stress Level: {latest_entry['StressLevel']}"
        )
        st.write(summary)

# Doctor Recommendation System

st.title("Doctor Recommendation System")

# Create a dictionary of doctors by specialization
doctors_dict = doctors_df.groupby('Specialization').apply(lambda df: df[['Name', 'Experience', 'Clinic Name', 'Fees', 'Address', 'Timing']].to_dict('records')).to_dict()

# Define question categories and their corresponding doctor types
question_to_doctor = {
    'Blood Pressure': 'General Physician',
    'Heart': 'Cardiologist',
    'Mental Health': 'Psychiatrist',
    'Ophthalmology': 'Ophthalmologist',
    'Orthopedics': 'Orthopedist',
    'Respiratory': 'Pulmonologist',
    'Neurology': 'Neurologist',
    'Dermatology': 'Dermatologist',
    'Gynecology': 'Gynecologist/Obstetrician',
    'Dental': 'Dentist',
    'Physiotherapy': 'Physiotherapist',
}

# Ask for user questions
user_query = st.text_area("Enter your health concern:")

# Process the user query
if user_query:
    st.write("Processing your query...")

    # Find relevant doctor based on the query
    found_doctor = False
    for keyword, doctor_type in question_to_doctor.items():
        if keyword.lower() in user_query.lower():
            if doctor_type in doctors_dict:
                st.write(f"We recommend consulting a {doctor_type}. Here are some doctors:")
                for doctor in doctors_dict[doctor_type]:
                    st.write(f"- Name: {doctor['Name']}")
                    st.write(f"  Experience: {doctor['Experience']}")
                    st.write(f"  Clinic: {doctor['Clinic Name']}")
                    st.write(f"  Fees: {doctor['Fees']}")
                    st.write(f"  Address: {doctor['Address']}")
                    st.write(f"  Timing: {doctor['Timing']}")
                    st.write("---")
                found_doctor = True
            break
    if not found_doctor:
        st.write("Sorry, I couldn't find a relevant doctor for your query.")

# Appointment Booking
selected_doctor = st.selectbox("Choose a doctor", [doctor['Name'] for doctor in doctors_df.to_dict('records')])
appointment_date = st.date_input("Select Appointment Date")
appointment_time = st.time_input("Select Appointment Time")

if st.button("Book Appointment"):
    st.success(f"Appointment booked with {selected_doctor} on {appointment_date} at {appointment_time}.")

# Emergency Contact
emergency_contact = st.text_input("Enter Emergency Contact Number")
if st.button("Send Emergency Alert"):
    # Simulate sending alert (function to send alert can be added here)
    st.success(f"Emergency alert sent to {emergency_contact}.")

# Ride Booking
st.subheader("Book a Ride to Your Appointment")
travel_options = ["Red Taxi", "Rapido"]
selected_travel = st.selectbox("Choose Travel Option", travel_options)

if st.button("Book Ride"):
    st.success(f"Ride booked with {selected_travel}.")

# Food Ordering
st.subheader("Order Food")
food_options = ["Swiggy", "Zomato"]
selected_food_service = st.selectbox("Choose Food Delivery Service", food_options)

if st.button("Order Food"):
    st.success(f"Order placed via {selected_food_service}.")

from twilio.rest import Client

# Twilio credentials
TWILIO_SID = 'AC333d68c1ba6f470d58d9efb24845a2d2'
TWILIO_AUTH_TOKEN = 'c40d224da730e1f438bf0c9cdaa80999'
TWILIO_PHONE_NUMBER = '919025740676'

# Create a Twilio client
client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# Emergency Contact
emergency_contact = st.text_input("Enter Emergency Contact Number")
if st.button("Send Emergency Alert"):
    # Send the alert
    try:
        message = client.messages.create(
            body="This is an emergency alert from your health monitoring system.",
            from_=TWILIO_PHONE_NUMBER,
            to=emergency_contact
        )
        st.success(f"Emergency alert sent to {emergency_contact}. Message SID: {message.sid}")
    except Exception as e:
        st.error(f"Failed to send emergency alert: {e}")
    except Exception as e:
        st.error(f"Failed to send emergency alert: {e}")

