#include <WiFi.h>
#include <ESP32Servo.h>   // library: "ESP32Servo"
#include <cstring>
#include <cstdio>

// ----------------------------
// Pins
// ----------------------------
#define PIN_STEP_L   33
#define PIN_DIR_L    19

#define PIN_STEP_R   25
#define PIN_DIR_R    18

#define PIN_EN       22    // set -1 if not used
#define PIN_SERVO    13

// ----------------------------
// WiFi
// ----------------------------
static const char* WIFI_SSID = "ANC_PLOTTER";
static const char* WIFI_PASS = "12345678";
static const uint16_t TCP_PORT = 3333;

// ----------------------------
// Driver / motion config
// ----------------------------
static const bool ENABLE_ACTIVE_LOW = true;
static const uint32_t STEP_PULSE_US = 3;

static const float FEED_TO_STEPS_PER_SEC = 0.6f;   // tuned
static const float MIN_STEPS_PER_SEC = 50.0f;
static const float MAX_STEPS_PER_SEC = 2500.0f;

static const int32_t MAX_STEPS_PER_CMD = 250000;

// ----------------------------
// Acceleration (new)
// ----------------------------
static const float START_STEPS_PER_SEC = 90.0f;       // start speed (lower to reduce jerk)
static const float ACCEL_STEPS_PER_SEC2 = 4000.0f;    // accel/decel (steps/s^2)

// ----------------------------
// Servo config
// ----------------------------
static const int SERVO_UP_DEG = 135;
static const int SERVO_DOWN_DEG = 55;
static const uint32_t SERVO_SETTLE_MS = 180;

// ----------------------------
// Communication
// ----------------------------
WiFiServer server(TCP_PORT);
WiFiClient tcpClient;

Servo penServo;
bool penIsDown = false;

int32_t posL = 0;
int32_t posR = 0;

// ----------------------------
// Queue
// ----------------------------
enum class Src : uint8_t { USB, NET };

struct Cmd {
  Src src;
  char line[160];
};

static const int QCAP = 64;
static Cmd q[QCAP];
static volatile int qHead = 0;
static volatile int qTail = 0;

static bool qPush(const Cmd& c) {
  int next = (qHead + 1) % QCAP;
  if (next == qTail) return false;
  q[qHead] = c;
  qHead = next;
  return true;
}

static bool qPop(Cmd& out) {
  if (qTail == qHead) return false;
  out = q[qTail];
  qTail = (qTail + 1) % QCAP;
  return true;
}

static inline void qClear() {
  qHead = 0;
  qTail = 0;
}

// ----------------------------
// Machine state + status
// ----------------------------
enum class MachineState : uint8_t { READY, BUSY, ERROR };

static MachineState machineState = MachineState::READY;
static char machineDetail[96] = "";

static void sendLineAll(const char* line) {
  Serial.print(line);
  Serial.print("\n");
  if (tcpClient && tcpClient.connected()) {
    tcpClient.print(line);
    tcpClient.print("\n");
  }
}

static void publishState(MachineState newState, const char* detail = nullptr) {
  bool sameState = (newState == machineState);
  bool sameDetail = false;
  if (detail == nullptr || detail[0] == 0) {
    sameDetail = (machineDetail[0] == 0);
  } else {
    sameDetail = (strcmp(machineDetail, detail) == 0);
  }

  if (sameState && sameDetail) return;

  machineState = newState;
  if (detail && detail[0]) {
    strncpy(machineDetail, detail, sizeof(machineDetail) - 1);
    machineDetail[sizeof(machineDetail) - 1] = 0;
  } else {
    machineDetail[0] = 0;
  }

  const char* label = (newState == MachineState::READY)
                          ? "READY"
                          : (newState == MachineState::BUSY ? "BUSY" : "ERROR");

  char buf[160];
  if (machineDetail[0]) {
    snprintf(buf, sizeof(buf), "STATE %s %s", label, machineDetail);
  } else {
    snprintf(buf, sizeof(buf), "STATE %s", label);
  }
  sendLineAll(buf);
}

static inline void setReady(const char* detail = nullptr) {
  publishState(MachineState::READY, detail);
}

static inline void setBusy(const char* detail) {
  publishState(MachineState::BUSY, detail);
}

static inline void setErrorState(const char* detail) {
  publishState(MachineState::ERROR, detail);
}

// ----------------------------
// Helpers
// ----------------------------
static inline void setEnable(bool enableMotors) {
#if PIN_EN >= 0
  if (ENABLE_ACTIVE_LOW) {
    digitalWrite(PIN_EN, enableMotors ? LOW : HIGH);
  } else {
    digitalWrite(PIN_EN, enableMotors ? HIGH : LOW);
  }
#else
  (void)enableMotors;
#endif
}

static inline void stepPulse(uint8_t pin) {
  digitalWrite(pin, HIGH);
  delayMicroseconds(STEP_PULSE_US);
  digitalWrite(pin, LOW);
}

static float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

static void penUp() {
  penServo.write(SERVO_UP_DEG);
  penIsDown = false;
  delay(SERVO_SETTLE_MS);
}

static void penDown() {
  penServo.write(SERVO_DOWN_DEG);
  penIsDown = true;
  delay(SERVO_SETTLE_MS);
}

static void replyOK(Src src) {
  if (src == Src::USB) {
    Serial.print("OK\n");
  } else {
    if (tcpClient && tcpClient.connected()) tcpClient.print("OK\n");
  }
}

static void replyERR(Src src, const char* msg) {
  if (src == Src::USB) {
    Serial.print("ERR ");
    Serial.print(msg);
    Serial.print("\n");
  } else {
    if (tcpClient && tcpClient.connected()) {
      tcpClient.print("ERR ");
      tcpClient.print(msg);
      tcpClient.print("\n");
    }
  }
}

static void reportError(Src src, const char* msg) {
  setErrorState(msg);
  replyERR(src, msg);
}

// Parse: "W L-123 R456 F1200"
static bool parseW(const char* s, int32_t& l, int32_t& r, int32_t& f) {
  const char* pL = strchr(s, 'L');
  const char* pR = strchr(s, 'R');
  const char* pF = strchr(s, 'F');
  if (!pL || !pR || !pF) return false;
  l = atoi(pL + 1);
  r = atoi(pR + 1);
  f = atoi(pF + 1);
  return true;
}

// ----------------------------
// Motion (Bresenham + acceleration)
// ----------------------------
static void moveWheels(int32_t lSteps, int32_t rSteps, int32_t feed) {
  if (lSteps == 0 && rSteps == 0) return;

  if (abs(lSteps) > MAX_STEPS_PER_CMD || abs(rSteps) > MAX_STEPS_PER_CMD) {
    return;
  }

  bool dirL = (lSteps >= 0);
  bool dirR = (rSteps >= 0);
  int32_t aL = abs(lSteps);
  int32_t aR = abs(rSteps);

  digitalWrite(PIN_DIR_L, dirL ? LOW : HIGH);
  // You reversed right DIR earlier, keep it here intentionally:
  digitalWrite(PIN_DIR_R, dirR ? HIGH : LOW);

  int32_t maxSteps = (aL > aR) ? aL : aR;
  if (maxSteps == 0) return;

  // target speed from feed
  float vTarget = clampf((float)feed * FEED_TO_STEPS_PER_SEC,
                         MIN_STEPS_PER_SEC,
                         MAX_STEPS_PER_SEC);

  // start speed (cannot exceed target)
  float v0 = START_STEPS_PER_SEC;
  if (v0 < MIN_STEPS_PER_SEC) v0 = MIN_STEPS_PER_SEC;
  if (v0 > vTarget) v0 = vTarget;

  // accel (must be > 0)
  float a = ACCEL_STEPS_PER_SEC2;
  if (a < 1.0f) a = 1.0f;

  // ramp length in steps: v^2 = v0^2 + 2*a*s
  float rampStepsF = (vTarget * vTarget - v0 * v0) / (2.0f * a);
  if (rampStepsF < 0) rampStepsF = 0;

  int32_t rampSteps = (int32_t)rampStepsF;

  // short move -> triangle profile
  int32_t half = maxSteps / 2;
  if (rampSteps > half) rampSteps = half;

  int32_t errL = 0;
  int32_t errR = 0;

  setEnable(true);

  uint32_t nextTick = micros();

  for (int32_t i = 0; i < maxSteps; i++) {
    int32_t into = i;
    int32_t left = (maxSteps - 1) - i;

    int32_t sRamp = into;
    if (left < sRamp) sRamp = left;

    float vNow = vTarget;

    if (rampSteps > 0 && sRamp < rampSteps) {
      vNow = sqrtf(v0 * v0 + 2.0f * a * (float)sRamp);
      if (vNow > vTarget) vNow = vTarget;
      if (vNow < v0) vNow = v0;
    }

    uint32_t period_us = (uint32_t)(1000000.0f / vNow);
    if (period_us < (STEP_PULSE_US + 2)) period_us = STEP_PULSE_US + 2;

    // wait for tick
    while ((int32_t)(micros() - nextTick) < 0) {
      delayMicroseconds(1);
    }
    nextTick = micros() + period_us;

    // Bresenham sync stepping
    errL += aL;
    if (errL >= maxSteps) {
      errL -= maxSteps;
      stepPulse(PIN_STEP_L);
      posL += dirL ? 1 : -1;
    }

    errR += aR;
    if (errR >= maxSteps) {
      errR -= maxSteps;
      stepPulse(PIN_STEP_R);
      posR += dirR ? 1 : -1;
    }
  }
}

// ----------------------------
// Command handling
// ----------------------------
static void handleLine(Src src, char* line) {
  // trim left
  while (*line == ' ' || *line == '\t') line++;

  // trim right
  int n = (int)strlen(line);
  while (n > 0 && (line[n - 1] == ' ' || line[n - 1] == '\t')) {
    line[n - 1] = 0;
    n--;
  }

  if (n == 0) {
    setReady();
    replyOK(src);
    return;
  }

  // END
  if (strcmp(line, "END") == 0) {
    setBusy("END");
    setEnable(false);
    setReady();
    replyOK(src);
    return;
  }

  // H
  if (strcmp(line, "H") == 0) {
    setBusy("HOME");
    posL = 0;
    posR = 0;
    setReady();
    replyOK(src);
    return;
  }

  // E 0 / E 1
  if (line[0] == 'E') {
    setBusy("ENABLE");
    if (strstr(line, "0")) {
      setEnable(false);
      setReady();
      replyOK(src);
      return;
    }
    if (strstr(line, "1")) {
      setEnable(true);
      setReady();
      replyOK(src);
      return;
    }
    reportError(src, "BAD_E");
    return;
  }

  // P U / P D
  if (line[0] == 'P') {
    setBusy("PEN");
    if (strstr(line, "U")) {
      penUp();
      setReady();
      replyOK(src);
      return;
    }
    if (strstr(line, "D")) {
      penDown();
      setReady();
      replyOK(src);
      return;
    }
    setReady();
    replyOK(src);
    return;
  }

  // W ...
  if (line[0] == 'W') {
    int32_t l, r, f;
    if (!parseW(line, l, r, f)) {
      reportError(src, "BAD_W");
      return;
    }
    setBusy("MOVE");
    moveWheels(l, r, f);
    setReady();
    replyOK(src);
    return;
  }

  reportError(src, "UNKNOWN");
}

// ----------------------------
// Line collectors
// ----------------------------
static char usbBuf[160];
static int usbLen = 0;

static char netBuf[160];
static int netLen = 0;

static inline void resetInputBuffers() {
  usbLen = 0;
  usbBuf[0] = 0;
  netLen = 0;
  netBuf[0] = 0;
}

static void pumpUSB() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      Cmd cmd;
      cmd.src = Src::USB;
      int copyLen = min(usbLen, (int)sizeof(cmd.line) - 1);
      memcpy(cmd.line, usbBuf, copyLen);
      cmd.line[copyLen] = 0;
      usbLen = 0;
      qPush(cmd);
      continue;
    }
    if (usbLen < (int)sizeof(usbBuf) - 1) {
      usbBuf[usbLen++] = c;
      usbBuf[usbLen] = 0;
    }
  }
}

static void pumpNET() {
  if (tcpClient && !tcpClient.connected()) {
    tcpClient.stop();
    resetInputBuffers();
    qClear();
    setEnable(false);
    setReady("TCP_LOST");
  }

  if (!tcpClient || !tcpClient.connected()) {
    WiFiClient nc = server.available();
    if (nc) {
      if (tcpClient) {
        tcpClient.stop();
      }
      tcpClient = nc;
      tcpClient.setNoDelay(true);
      resetInputBuffers();
      qClear();
      setEnable(false);
      setReady("TCP_CONNECTED");
    }
    return;
  }

  while (tcpClient.available()) {
    char c = (char)tcpClient.read();
    if (c == '\r') continue;
    if (c == '\n') {
      Cmd cmd;
      cmd.src = Src::NET;
      int copyLen = min(netLen, (int)sizeof(cmd.line) - 1);
      memcpy(cmd.line, netBuf, copyLen);
      cmd.line[copyLen] = 0;
      netLen = 0;
      qPush(cmd);
      continue;
    }
    if (netLen < (int)sizeof(netBuf) - 1) {
      netBuf[netLen++] = c;
      netBuf[netLen] = 0;
    }
  }
}

// ----------------------------
// Setup / loop
// ----------------------------
void setup() {
  Serial.begin(115200);

  pinMode(PIN_STEP_L, OUTPUT);
  pinMode(PIN_DIR_L, OUTPUT);
  pinMode(PIN_STEP_R, OUTPUT);
  pinMode(PIN_DIR_R, OUTPUT);

#if PIN_EN >= 0
  pinMode(PIN_EN, OUTPUT);
#endif

  digitalWrite(PIN_STEP_L, LOW);
  digitalWrite(PIN_STEP_R, LOW);
  setEnable(false);

  penServo.setPeriodHertz(50);
  penServo.attach(PIN_SERVO, 500, 2400);
  penUp();

  WiFi.mode(WIFI_AP);
  WiFi.softAP(WIFI_SSID, WIFI_PASS);

  server.begin();
  server.setNoDelay(true);

  publishState(MachineState::READY, "BOOT");
  Serial.print("OK\n");
}

void loop() {
  pumpUSB();
  pumpNET();

  Cmd cmd;
  if (qPop(cmd)) {
    handleLine(cmd.src, cmd.line);
  }

  delay(1);
}
