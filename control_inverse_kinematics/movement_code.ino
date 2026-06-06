#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVO_MIN  150 
#define SERVO_MAX  600 

// =================================================================
// 🔌 YOUR VERIFIED PORT MAPPING
// =================================================================
int L_HIP_PORT   = 14;
int L_KNEE_PORT  = 13;
int L_ANKLE_PORT = 12;

int R_HIP_PORT   = 2;
int R_KNEE_PORT  = 1;
int R_ANKLE_PORT = 0;

// =================================================================
// 🎯 YOUR TRUE ZERO OFFSETS (CALIBRATED)
// =================================================================
int hipLOffset   = 90;
int kneeLOffset  = 90;
int ankleLOffset = 90;

int hipROffset   = 90;
int kneeROffset  = 90;
int ankleROffset = 90;

// =================================================================
// 🛠️ GAIT TUNING SANDBOX (CHANGE THESE VALUES!)
// =================================================================
float stepHeight    = 10.2;  // Straighter legs (Max physical length is 10.7)
float leanAmount    = 1.2;   // Reduced side-to-side "pending" sway
float stride        = 2.0;   // Moderate step length
float stepClearance = 1.0;   // Moderate foot lift

int   swingSpeed    = 25;    // Speed of stepping forward (Higher = Slower)
int   leanSpeed     = 35;    // Speed of weight shifting (Higher = Slower)
int   pauseBetween  = 200;   // Pause in ms between taking left/right steps

// Physical hardware measurements
float l1 = 5.0;
float l2 = 5.7;

// Safely translate degrees to PCA9685 pulses
int degreeToPulse(int degree) {
  if (degree < 0)   degree = 0;
  if (degree > 180) degree = 180;
  return map(degree, 0, 180, SERVO_MIN, SERVO_MAX);
}

// Write calculated angles to the specific hardware offsets
void updateServoPos(int target1, int target2, int target3, char leg) {
  if (leg == 'l') {
    pwm.setPWM(L_HIP_PORT,   0, degreeToPulse(hipLOffset   - target1));
    pwm.setPWM(L_KNEE_PORT,  0, degreeToPulse(kneeLOffset  - target2));
    pwm.setPWM(L_ANKLE_PORT, 0, degreeToPulse(2 * ankleLOffset - target3));
  } else if (leg == 'r') {
    pwm.setPWM(R_HIP_PORT,   0, degreeToPulse(hipROffset   + target1));
    pwm.setPWM(R_KNEE_PORT,  0, degreeToPulse(kneeROffset  + target2));
    pwm.setPWM(R_ANKLE_PORT, 0, degreeToPulse(target3));
  }
}

// Inverse Kinematics Math Generator
void pos(float x, float z, char leg) {
  // Fix for backwards walking
  x = -x; 

  float hipRad2 = atan(x / z);
  float hipDeg2 = hipRad2 * (180 / PI);
  float z2      = z / cos(hipRad2);
  
  if (z2 > (l1 + l2)) z2 = l1 + l2 - 0.01; 
  
  float hipRad1 = acos((sq(l1) + sq(z2) - sq(l2)) / (2 * l1 * z2));
  float hipDeg1 = hipRad1 * (180 / PI);
  float kneeRad  = PI - acos((sq(l1) + sq(l2) - sq(z2)) / (2 * l1 * l2));
  float ankleRad = PI / 2 + hipRad2 - acos((sq(l2) + sq(z2) - sq(l1)) / (2 * l2 * z2));
  
  float hipDeg   = hipDeg1 + hipDeg2;
  float kneeDeg  = kneeRad  * (180 / PI);
  float ankleDeg = ankleRad * (180 / PI);
  
  updateServoPos(hipDeg, kneeDeg, ankleDeg, leg);
}

// Upgraded Dynamic Step Function
void takeDynamicStep(char leadLeg) {
  char supportLeg = (leadLeg == 'l') ? 'r' : 'l';

  // PHASE 1: LEAN (Shift weight)
  for (float x = 0; x <= leanAmount; x += 0.2) {
    pos(-x, stepHeight, leadLeg);
    pos(-x, stepHeight, supportLeg);
    delay(leanSpeed);
  }

  // PHASE 2: FALL & SWING
  for (float i = 0; i <= stride; i += 0.5) {
    pos(-(leanAmount + i), stepHeight, supportLeg); 
    pos(i, stepHeight - stepClearance, leadLeg);    
    delay(swingSpeed);
  }

  // PHASE 3: CATCH
  pos(stride, stepHeight, leadLeg);
  delay(50);

  // PHASE 4: RECOVER 
  for (float i = -(leanAmount + stride); i <= 0; i += 0.5) {
    pos(stride + i, stepHeight, leadLeg);
    pos(i, stepHeight - (stepClearance / 2), supportLeg); 
    delay(swingSpeed);
  }

  // Reset to center
  pos(0, stepHeight, 'l');
  pos(0, stepHeight, 'r');
  delay(50);
}

void initialize() {
  for (float i = 10.9; i >= stepHeight; i -= 0.1) {
    pos(0, i, 'l');
    pos(0, i, 'r');
    delay(20); 
  }
}

void setup() {
  Serial.begin(115200); 
  Wire.begin();
  Wire.setClock(400000); 
  
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(50);
  delay(10);

  Serial.println("Waking up servos safely...");
  
  // Staggered power-on
  pwm.setPWM(L_HIP_PORT,   0, degreeToPulse(hipLOffset));   delay(200);
  pwm.setPWM(L_KNEE_PORT,  0, degreeToPulse(kneeLOffset));  delay(200);
  pwm.setPWM(L_ANKLE_PORT, 0, degreeToPulse(ankleLOffset)); delay(200);
  
  pwm.setPWM(R_HIP_PORT,   0, degreeToPulse(hipROffset));   delay(200);
  pwm.setPWM(R_KNEE_PORT,  0, degreeToPulse(kneeROffset));  delay(200);
  pwm.setPWM(R_ANKLE_PORT, 0, degreeToPulse(ankleROffset)); delay(200);

  Serial.println("Initial stance locked. Hold robot in the air...");
  delay(5000); // 5 seconds to set it on the desk
  
  initialize(); 
  Serial.println("Ready to walk!");
}

void loop() {
  takeDynamicStep('r'); 
  delay(pauseBetween);  
  takeDynamicStep('l'); 
  delay(pauseBetween);  
}