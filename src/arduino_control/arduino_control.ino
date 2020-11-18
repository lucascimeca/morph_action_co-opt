#include <Servo.h>
#include <string.h>

const int Apwm = 3 ;  //initializing pin 2 as pwm
const int Bpwm = 9 ; 
const int Ain_1 = 2 ;
const int Ain_2 = 4 ;
const int Bin_1 = 12 ;
const int Bin_2 = 8 ;
const int STBY = 7 ;

String inByte;
int int_cmd;
char char_cmd[1];
char char_cmd2[3];
char cmds[100];
char * cmd;

void setup() {
  pinMode(Apwm, OUTPUT) ;   //we have to set PWM pin as output
  pinMode(Bpwm, OUTPUT) ;   
  pinMode(Ain_1, OUTPUT) ;  //Logic pins are also set as output
  pinMode(Ain_2, OUTPUT) ;
  pinMode(Bin_1, OUTPUT) ; 
  pinMode(Bin_2, OUTPUT) ;
  pinMode(STBY, OUTPUT) ;
  Serial.begin(9600);
}

// can send commands of type " 'f:10', 'f:55', 'f:150', 'f255' etc. or 'b:46', 'b:130', etc, or just 's' to stop motors"
void loop(){   
  if(Serial.available()){  // if data available in serial port 
    inByte = Serial.readStringUntil('\n'); // read data until newline

    // initialize variables, so can detect command errors
    int_cmd = -1;
    char_cmd[0] = 'z';
    
    inByte.toCharArray(cmds, 100);
    cmd = strtok(cmds, ":");
    while (cmd != NULL){
      if (!isDigit(cmd[0])) {
        char_cmd[0] = cmd[0]; 
      } else if (isDigit(cmd[0])) {
        int_cmd = atoi(cmd);
      }
      cmd = strtok(0, ":");
    }

    if (int_cmd != -1 || char_cmd[0] == 's') {
      switch (char_cmd[0]) {
        case 'f':
          analogWrite(Apwm,  int_cmd) ;
          digitalWrite(STBY, HIGH);
          digitalWrite(Ain_1, HIGH);
          digitalWrite(Ain_2, LOW);
          digitalWrite(Bin_1, HIGH);
          digitalWrite(Bin_2, HIGH);
          Serial.println("1");
          break;
        case 'b':
          analogWrite(Bpwm,  int_cmd) ;
          digitalWrite(STBY, HIGH);
          digitalWrite(Ain_1, HIGH);
          digitalWrite(Ain_2, HIGH);
          digitalWrite(Bin_1, LOW);
          digitalWrite(Bin_2, HIGH);
          Serial.println("1");
          break;
        case 's':
          digitalWrite(STBY, LOW);
          digitalWrite(Ain_1, HIGH);
          digitalWrite(Ain_2, HIGH);
          digitalWrite(Bin_1, HIGH);
          digitalWrite(Bin_2, HIGH);
          Serial.println("1");
          break;
        default:
          Serial.println("0");
          break;
      }
     }
     else {
      Serial.println("0");
     }
   }
}
