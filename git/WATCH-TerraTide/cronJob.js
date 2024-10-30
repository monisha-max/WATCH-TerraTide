// const cron = require('node-cron');
// const nodemailer = require('nodemailer');
// const { MongoClient } = require('mongodb');
// const axios = require('axios');

// // MongoDB connection (Update with MongoDB Atlas URI)
// const uri = 'mongodb+srv://terratide662:Nk50opwMr51Ig1Ra@cluster0.exm1e.mongodb.net/?retryWrites=true&w=majority';
// const client = new MongoClient(uri, { useNewUrlParser: true, useUnifiedTopology: true });

// const API_KEY = 'bf1a8422d92b4eedb2d143836242408';  // Weather API Key

// // Function to send email alerts
// async function sendEmail(userEmail, message) {
//     let transporter = nodemailer.createTransport({
//         service: 'gmail',
//         auth: {
//             user: 'terratide662@gmail.com',
//             pass: 'Monisha$1234',
//         },
//     });

//     let mailOptions = {
//         from: 'terratide662@gmail.com',
//         to: userEmail,
//         subject: 'Heatwave Alert',
//         text: message,
//     };

//     try {
//         await transporter.sendMail(mailOptions);
//         console.log(`Email sent to ${userEmail}`);
//     } catch (error) {
//         console.error(`Error sending email to ${userEmail}:`, error);
//     }
// }

// // Function to check the weather and send alerts
// async function checkWeatherAndSendAlerts() {
//     try {
//         await client.connect();
//         const database = client.db('heat_tolerance_db');
//         const usersCollection = database.collection('users');
//         const users = await usersCollection.find().toArray();

//         for (let user of users) {
//             const city = user.city;
//             const response = await axios.get(`http://api.weatherapi.com/v1/current.json?key=${API_KEY}&q=${city}&aqi=no`);
//             const weather = response.data.current;
//             const temp = weather.temp_c;
//             const humidity = weather.humidity;
//             const rainfall = weather.precip_mm;

//             // Customize logic based on health conditions
//             let alertMessage = null;

//             if (user.health_conditions === 'High Blood Pressure' && temp > 30) {
//                 alertMessage = `
//                     Hi ${user.name},
            
//                     The temperature is currently ${temp}°C in ${city}. 
//                     As someone with high blood pressure, it's recommended that you avoid going outside during high heat. 
//                     High temperatures can increase your risk of dehydration and cause additional strain on your heart.
            
//                     Here are some tips to stay safe:
//                     - Stay indoors as much as possible, especially during peak heat hours.
//                     - Drink plenty of water throughout the day, and avoid alcohol and caffeine.
//                     - Wear loose, breathable clothing if you must go out.
            
//                     Take care and stay cool!
            
//                     Best regards,
//                     ClimaBot Team
//                 `;
//             } else if (user.health_conditions === 'Asthma' && humidity > 80) {
//                 alertMessage = `
//                     Hi ${user.name},
            
//                     The humidity level is currently ${humidity}% in ${city}. 
//                     As someone with asthma, it's important to be cautious when humidity levels are high. 
//                     Humid air can make it harder to breathe and may trigger your symptoms.
            
//                     Here's what you can do:
//                     - Avoid outdoor activities during peak humidity.
//                     - Stay indoors with air conditioning or a dehumidifier if possible.
//                     - Carry your inhaler with you at all times.
            
//                     Breathe easy, and stay safe!
            
//                     Warm regards,
//                     ClimaBot Team
//                 `;
//             } else if (user.activity_level === 'high' && temp > 35) {
//                 alertMessage = `
//                     Hi ${user.name},
            
//                     The temperature in ${city} is currently ${temp}°C, and it's a heatwave! 
//                     Since you indicated a high activity level, please take extra care in these conditions.
            
//                     We suggest:
//                     - Postpone any outdoor activities during the hottest parts of the day.
//                     - Drink plenty of fluids before, during, and after your workout.
//                     - Wear a hat and apply sunscreen if you must be outdoors.
//                     - Consider doing indoor activities or exercise in a cool place instead.
            
//                     Your health comes first, so stay safe during the heat!
            
//                     Cheers,
//                     ClimaBot Team
//                 `;
//             } else if (user.health_conditions === 'Heart Disease' && rainfall > 5 && temp > 25) {
//                 alertMessage = `
//                     Hi ${user.name},
            
//                     There's currently a temperature of ${temp}°C and rain in ${city}. As someone with a heart condition, 
//                     these weather conditions could be risky for outdoor activities.
            
//                     What you can do to stay safe:
//                     - Avoid strenuous activity in this weather.
//                     - Stay hydrated and monitor your heart rate.
//                     - Wear lightweight and moisture-wicking clothing to stay cool and dry.
            
//                     Take care of yourself and consider resting indoors today.
            
//                     Best,
//                     ClimaBot Team
//                 `;
//             } else if (temp <= 30 && humidity <= 70 && rainfall === 0) {
//                 alertMessage = `
//                     Hi ${user.name},
            
//                     Good news! The weather in ${city} is currently clear and safe for outdoor activities. 
//                     The temperature is a comfortable ${temp}°C, and humidity levels are manageable at ${humidity}%.
            
//                     If you're planning to head outdoors, here are a few tips to stay healthy:
//                     - Stay hydrated, even in comfortable weather.
//                     - Wear light and breathable clothing.
//                     - Apply sunscreen if you'll be in the sun for an extended period.
            
//                     Enjoy your time outside and have a great day!
            
//                     Regards,
//                     ClimaBot Team
//                 `;
//             } else if (temp < 10 && user.health_conditions === 'Asthma') {
//                 alertMessage = `
//                     Hi ${user.name},
            
//                     The temperature in ${city} has dropped to ${temp}°C. 
//                     Cold weather can sometimes exacerbate asthma symptoms, making it harder to breathe.
            
//                     Here are some ways to protect yourself:
//                     - Wear a scarf or mask over your nose and mouth when going outside.
//                     - Warm up your body with light exercises before stepping out.
//                     - Keep your rescue inhaler nearby.
            
//                     Stay warm and take it easy!
            
//                     Best regards,
//                     ClimaBot Team
//                 `;
//             } else if (user.age >= 65 && temp > 35) {
//                 alertMessage = `
//                     Hi ${user.name},
            
//                     The temperature in ${city} is currently ${temp}°C, and it may pose risks, especially for older adults. 
//                     Since you are in a more sensitive age group, please take extra precautions during these extreme heat conditions.
            
//                     Tips for staying cool:
//                     - Stay indoors during the hottest part of the day (10 AM to 4 PM).
//                     - Drink plenty of water even if you don’t feel thirsty.
//                     - Use fans, air conditioning, or cool showers to stay cool.
            
//                     Your health is important, so stay safe and hydrated!
            
//                     Best regards,
//                     ClimaBot Team
//                 `;
//             }

//             if (alertMessage) {
//                 await sendEmail(user.email, alertMessage);
//             }
//         }
//     } finally {
//         await client.close();
//     }
// }

// // Schedule the cron job to run every hour
// //cron.schedule('0 * * * *', checkWeatherAndSendAlerts);
// cron.schedule('53 9 * * *', checkWeatherAndSendAlerts);
const cron = require('node-cron');
const nodemailer = require('nodemailer');
const { MongoClient } = require('mongodb');
const axios = require('axios');

// MongoDB connection (Update with MongoDB Atlas URI)
const uri = 'mongodb+srv://terratide662:Nk50opwMr51Ig1Ra@cluster0.exm1e.mongodb.net/?retryWrites=true&w=majority';
const client = new MongoClient(uri, { useNewUrlParser: true, useUnifiedTopology: true });

const API_KEY = 'bf1a8422d92b4eedb2d143836242408';  // Weather API Key

// Function to send email alerts
async function sendEmail(userEmail, message) {
    let transporter = nodemailer.createTransport({
        service: 'gmail',
        auth: {
            user: 'terratide662@gmail.com',
            pass: 'wlfo iiqi eoeo iyce',  // Make sure this is correct or use an app password
        },
    });

    let mailOptions = {
        from: 'terratide662@gmail.com',
        to: userEmail,
        subject: 'Heatwave Alert',
        text: message,
    };

    try {
        await transporter.sendMail(mailOptions);
        console.log(`Email sent to ${userEmail}`);
    } catch (error) {
        console.error(`Error sending email to ${userEmail}:`, error);
    }
}

// Function to check the weather and send alerts
async function checkWeatherAndSendAlerts() {
    try {
        console.log('Connecting to MongoDB...');
        await client.connect();
        const database = client.db('heat_tolerance_db');
        const usersCollection = database.collection('users');
        const users = await usersCollection.find().toArray();
        
        console.log('Users fetched from database:', users);

        for (let user of users) {
            const city = user.city;
            console.log(`Fetching weather for city: ${city}`);
            const response = await axios.get(`http://api.weatherapi.com/v1/current.json?key=${API_KEY}&q=${city}&aqi=no`);
            const weather = response.data.current;
            const temp = weather.temp_c;
            const humidity = weather.humidity;
            const rainfall = weather.precip_mm;

            console.log(`Weather data for ${city}: temp=${temp}, humidity=${humidity}, rainfall=${rainfall}`);

            // Customize logic based on health conditions
            let alertMessage = null;

            // (Condition checks and message generation here)
            if (user.health_conditions === 'High Blood Pressure' && temp > 30) {
                                alertMessage = `
                                    Hi ${user.name},
                            
                                    The temperature is currently ${temp}°C in ${city}. 
                                    As someone with high blood pressure, it's recommended that you avoid going outside during high heat. 
                                    High temperatures can increase your risk of dehydration and cause additional strain on your heart.
                            
                                    Here are some tips to stay safe:
                                    - Stay indoors as much as possible, especially during peak heat hours.
                                    - Drink plenty of water throughout the day, and avoid alcohol and caffeine.
                                    - Wear loose, breathable clothing if you must go out.
                            
                                    Take care and stay cool!
                            
                                    Best regards,
                                    ClimaBot Team
                                `;
                            } else if (user.health_conditions === 'Asthma' && humidity > 80) {
                                alertMessage = `
                                    Hi ${user.name},
                            
                                    The humidity level is currently ${humidity}% in ${city}. 
                                    As someone with asthma, it's important to be cautious when humidity levels are high. 
                                    Humid air can make it harder to breathe and may trigger your symptoms.
                            
                                    Here's what you can do:
                                    - Avoid outdoor activities during peak humidity.
                                    - Stay indoors with air conditioning or a dehumidifier if possible.
                                    - Carry your inhaler with you at all times.
                            
                                    Breathe easy, and stay safe!
                            
                                    Warm regards,
                                    ClimaBot Team
                                `;
                            } else if (user.activity_level === 'high' && temp > 35) {
                                alertMessage = `
                                    Hi ${user.name},
                            
                                    The temperature in ${city} is currently ${temp}°C, and it's a heatwave! 
                                    Since you indicated a high activity level, please take extra care in these conditions.
                            
                                    We suggest:
                                    - Postpone any outdoor activities during the hottest parts of the day.
                                    - Drink plenty of fluids before, during, and after your workout.
                                    - Wear a hat and apply sunscreen if you must be outdoors.
                                    - Consider doing indoor activities or exercise in a cool place instead.
                            
                                    Your health comes first, so stay safe during the heat!
                            
                                    Cheers,
                                    ClimaBot Team
                                `;
                            } else if (user.health_conditions === 'Heart Disease' && rainfall > 5 && temp > 25) {
                                alertMessage = `
                                    Hi ${user.name},
                            
                                    There's currently a temperature of ${temp}°C and rain in ${city}. As someone with a heart condition, 
                                    these weather conditions could be risky for outdoor activities.
                            
                                    What you can do to stay safe:
                                    - Avoid strenuous activity in this weather.
                                    - Stay hydrated and monitor your heart rate.
                                    - Wear lightweight and moisture-wicking clothing to stay cool and dry.
                            
                                    Take care of yourself and consider resting indoors today.
                            
                                    Best,
                                    ClimaBot Team
                                `;
                            } else if (temp <= 30 && humidity <= 70 && rainfall === 0) {
                                alertMessage = `
                                    Hi ${user.name},
                            
                                    Good news! The weather in ${city} is currently clear and safe for outdoor activities. 
                                    The temperature is a comfortable ${temp}°C, and humidity levels are manageable at ${humidity}%.
                            
                                    If you're planning to head outdoors, here are a few tips to stay healthy:
                                    - Stay hydrated, even in comfortable weather.
                                    - Wear light and breathable clothing.
                                    - Apply sunscreen if you'll be in the sun for an extended period.
                            
                                    Enjoy your time outside and have a great day!
                            
                                    Regards,
                                    ClimaBot Team
                                `;
                            } else if (temp < 10 && user.health_conditions === 'Asthma') {
                                alertMessage = `
                                    Hi ${user.name},
                            
                                    The temperature in ${city} has dropped to ${temp}°C. 
                                    Cold weather can sometimes exacerbate asthma symptoms, making it harder to breathe.
                            
                                    Here are some ways to protect yourself:
                                    - Wear a scarf or mask over your nose and mouth when going outside.
                                    - Warm up your body with light exercises before stepping out.
                                    - Keep your rescue inhaler nearby.
                            
                                    Stay warm and take it easy!
                            
                                    Best regards,
                                    ClimaBot Team
                                `;
                            } else if (user.age >= 65 && temp > 35) {
                                alertMessage = `
                                    Hi ${user.name},
                            
                                    The temperature in ${city} is currently ${temp}°C, and it may pose risks, especially for older adults. 
                                    Since you are in a more sensitive age group, please take extra precautions during these extreme heat conditions.
                            
                                    Tips for staying cool:
                                    - Stay indoors during the hottest part of the day (10 AM to 4 PM).
                                    - Drink plenty of water even if you don’t feel thirsty.
                                    - Use fans, air conditioning, or cool showers to stay cool.
                            
                                    Your health is important, so stay safe and hydrated!
                            
                                    Best regards,
                                    ClimaBot Team
                                `;
                            }

            if (alertMessage) {
                console.log(`Sending alert to ${user.email}`);
                await sendEmail(user.email, alertMessage);
            } else {
                const allGoodMessage = `
                    Hi ,
                    
                    The weather in ${city} is currently safe and there are no specific alerts for your health conditions.
                    Feel free to continue with your day as planned, but always stay hydrated and take care!

                    Best regards,
                    ClimaBot Team
                `;
                console.log(`Sending 'All Good to Go' message to ${user.email}`);
                await sendEmail(user.email, allGoodMessage);
            }
        }
    } catch (error) {
        console.error('Error in checkWeatherAndSendAlerts:', error);
    } finally {
        await client.close();
        console.log('MongoDB connection closed.');
    }
}

// Schedule the cron job to run every hour
cron.schedule('2 * * * *', checkWeatherAndSendAlerts);  // Run every minute for testing


