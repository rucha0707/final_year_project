#include <Servo.h>

#define SAMPLE_RATE 2000
#define BAUD_RATE 115200

#define SHIELD_PIN A0
#define PILL_PIN   A2
#define SERVO_PIN  2 // NEW: Muscle shield servo pin

#define BUFFER_SIZE 128

Servo myServo; // NEW
int circular_buffer[BUFFER_SIZE];
int data_index = 0;
int sum = 0;

void setup() {
  Serial.begin(BAUD_RATE);
  myServo.attach(SERVO_PIN); // NEW
  myServo.write(0);          // NEW: Start at 0
}

void loop() {
  // NEW: Read prediction back from Python
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == '0') {
      myServo.write(180); // Hand Open
    } else if (command == '1') {
      myServo.write(90);  // Hand Close
    }
  }

  static unsigned long past = 0;
  unsigned long present = micros();
  unsigned long interval = present - past;
  past = present;

  static long timer = 0;
  timer -= interval;

  if (timer < 0) {
    timer += 1000000 / SAMPLE_RATE;

    float shield_raw = analogRead(SHIELD_PIN);
    float pill_raw   = analogRead(PILL_PIN);

    float shield_filtered = EMGFilter(shield_raw);
    float pill_filtered   = EMGFilter(pill_raw);

    int shield_scaled = shield_filtered * 1000;
    int pill_scaled   = pill_filtered * 1000;

    int shield_envelope = getEnvelope(abs(shield_scaled));

    float out1 = shield_envelope / 1000.0;
    float out2 = pill_scaled / 1000.0;

    Serial.print(out1, 6);
    Serial.print(",");
    Serial.println(out2, 6);
  }
}

int getEnvelope(int abs_emg) {
  sum -= circular_buffer[data_index];
  sum += abs_emg;
  circular_buffer[data_index] = abs_emg;
  data_index = (data_index + 1) % BUFFER_SIZE;
  return (sum / BUFFER_SIZE);
}

float EMGFilter(float input) {
  float output = input;
  {
    static float z1, z2;
    float x = output - 0.05159732*z1 - 0.36347401*z2;
    output = 0.01856301*x + 0.03712602*z1 + 0.01856301*z2;
    z2 = z1; z1 = x;
  }
  {
    static float z1, z2;
    float x = output - -0.53945795*z1 - 0.39764934*z2;
    output = x - 2.0*z1 + z2;
    z2 = z1; z1 = x;
  }
  {
    static float z1, z2;
    float x = output - 0.47319594*z1 - 0.70744137*z2;
    output = x + 2.0*z1 + z2;
    z2 = z1; z1 = x;
  }
  {
    static float z1, z2;
    float x = output - -1.00211112*z1 - 0.74520226*z2;
    output = x - 2.0*z1 + z2;
    z2 = z1; z1 = x;
  }
  return output;
}