// Code to go on Barco Polo's Arduino. Gets PWM values for each motor from a serial connection.
// Writes the PWM value to each motor accordingly.

// NOTES: This code is currently untested. It requires double-checking of servo pins.


# include <Servo.h>

// Define the servo pins. Please note that this may change based on the hardware.
// If referencing motor_mix.ino (RoboBoat 2024), 
// [servo1, servo 2]: stern
// [servo3, servo4]: aft
// [servo3, (probably) servo1]: port
// [servo4, (probably) servo2]: starboard

byte servoPin_stern_port = 3;
byte servoPin_stern_starboard = 5;
byte servoPin_aft_port = 6;
byte servoPin_aft_starboard = 9;

// Define servo objects.
Servo servo_stern_port;
Servo servo_stern_starboard;
Servo servo_aft_port;
Servo servo_aft_starboard;

int stern_port_PWM = 1500;
int stern_starboard_PWM = 1500;
int aft_port_PWM = 1500;
int aft_starboard_PWM = 1500;

// Setup function (runs during initialization)
void setup(){
    // Initialize Serial Communication at 9600 Baud Rate
    Serial.begin(9600);

    // Attach the right servo object to the right pin
    servo_stern_port.attach(servoPin_stern_port);
    servo_stern_starboard.attach(servoPin_stern_starboard);
    servo_aft_port.attach(servoPin_aft_port);
    servo_aft_starboard.attach(servoPin_aft_starboard);

    // Write initial PWM values to the servos (motors)
    servo_stern_port.write(1500);
    servo_stern_starboard.write(1500);
    servo_aft_port.write(1500);
    servo_aft_starboard.write(1500);

    delay(1000);
}

void parseData(String data){
    // NOTE: Can make it more graceful later.
    // Finds the each comma in the data, stores the value before the comma in a variable

    // Find index of comma
    int commaIndex1 = data.indexOf(',');
    int commaIndex2 = data.indexOf(',', commaIndex1 + 1);
    int commaIndex3 = data.indexOf(',', commaIndex2 + 1);

    // Extract value, convert it to an integer from a string
    stern_port_PWM = data.substring(0, commaIndex1).toInt();
    stern_starboard_PWM = data.substring(commaIndex1 + 1, commaIndex2).toInt();
    aft_port_PWM = data.substring(commaIndex2 + 1, commaIndex3).toInt();
    aft_starboard_PWM = data.substring(commaIndex3 + 1).toInt();
}

void sendMotorCommands(){
    servo_stern_port.write(stern_port_PWM);
    servo_stern_starboard.write(stern_starboard_PWM);
    servo_aft_port.write(aft_port_PWM);
    servo_aft_starboard.write(aft_starboard_PWM);
}

void debugDisplay(){
    Serial.print("PWM Value[Stern Port]: ");
    Serial.println(stern_port_PWM);
    Serial.print("PWM Value[Stern Starboard]: ");
    Serial.println(stern_starboard_PWM);
    Serial.print("PWM Value[Aft Port]: ");
    Serial.println(aft_port_PWM);
    Serial.print("PWM Value[Aft Starboard]: ");
    Serial.println(aft_starboard_PWM);
}

void loop(){
    // Check for serial commands and write those commands to the servo.
    // The commands should be in the form of a list: [stern_port, stern_starboard, aft_port, aft_starboard]
    
    if (Serial.available){
        receivedData = Serial.readStringUntil('\n');
        parseData(receivedData);
        debugDisplay();
        delay(250);
    }
}