#!/usr/bin/env python3
""" Simulator connector for FlightAxis Link

Copyright (C) 2019  BioBrain, Inc

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import re
import requests
import socket
import time
import numpy as np

from xml.etree import ElementTree as ET

SIM_DEFAULT = { # (name, value, save)        
        "AHRS_EKF_TYPE": (10, False),
        "INS_GYR_CAL": (0, False),
        "RC1_MIN": (1000, True),
        "RC1_MAX": (2000, True),
        "RC2_MIN": (1000, True),
        "RC2_MAX": (2000, True),
        "RC3_MIN": (1000, True),
        "RC3_MAX": (2000, True),
        "RC4_MIN": (1000, True),
        "RC4_MAX": (2000, True),
        "RC2_REVERSED": (1, False), # interlink has reversed rc2
        "SERVO1_MIN": (1000, False),
        "SERVO1_MAX": (2000, False),
        "SERVO2_MIN": (1000, False),
        "SERVO2_MAX": (2000, False),
        "SERVO3_MIN": (1000, False),
        "SERVO3_MAX": (2000, False),
        "SERVO4_MIN": (1000, False),
        "SERVO4_MAX": (2000, False),
        "SERVO5_MIN": (1000, False),
        "SERVO5_MAX": (2000, False),
        "SERVO6_MIN": (1000, False),
        "SERVO6_MAX": (2000, False),
        "SERVO6_MIN": (1000, False),
        "SERVO6_MAX": (2000, False),
        "INS_ACC2OFFS_X": (0.001, False),
        "INS_ACC2OFFS_Y": (0.001, False),
        "INS_ACC2OFFS_Z": (0.001, False),
        "INS_ACC2SCAL_X": (1.001, False),
        "INS_ACC2SCAL_Y": (1.001, False),
        "INS_ACC2SCAL_Z": (1.001, False),
        "INS_ACCOFFS_X": (0.001, False),
        "INS_ACCOFFS_Y": (0.001, False),
        "INS_ACCOFFS_Z": (0.001, False),
        "INS_ACCSCAL_X": (1.001, False),
        "INS_ACCSCAL_Y": (1.001, False),
        "INS_ACCSCAL_Z": (1.001, False),
}

ACTION_FMT = {
    "RestoreOriginalControllerDevice":
    ("<?xml version='1.0' encoding='UTF-8'?>\n"
     "<soap:Envelope xmlns:soap='http://schemas.xmlsoap.org/soap/envelope/' "
     "xmlns:xsd='http://www.w3.org/2001/XMLSchema' "
     "xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>\n"
     "<soap:Body>\n"
     "<RestoreOriginalControllerDevice><a>1</a><b>2</b>"
     "</RestoreOriginalControllerDevice>\n"
     "</soap:Body>\n"
     "</soap:Envelope>"),
    "InjectUAVControllerInterface":
    ("<?xml version='1.0' encoding='UTF-8'?> "
     "<soap:Envelope xmlns:soap='http://schemas.xmlsoap.org/soap/envelope/' "
     "xmlns:xsd='http://www.w3.org/2001/XMLSchema' "
     "xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'> "
     "<soap:Body> "
     "<InjectUAVControllerInterface><a>1</a><b>2</b>"
     "</InjectUAVControllerInterface> "
     "</soap:Body> "
     "</soap:Envelope>"),
    "ExchangeData":
    ("<?xml version='1.0' encoding='UTF-8'?> "
     "<soap:Envelope xmlns:soap='http://schemas.xmlsoap.org/soap/envelope/' "
     "xmlns:xsd='http://www.w3.org/2001/XMLSchema' "
     "xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'> "
     "<soap:Body> "
     "<ExchangeData> "
     "<pControlInputs> "
     "<m-selectedChannels>4095</m-selectedChannels> "
     "<m-channelValues-0to1> "
     "<item>{:.4f}</item> "
     "<item>{:.4f}</item> "
     "<item>{:.4f}</item> "
     "<item>{:.4f}</item> "
     "<item>{:.4f}</item> "
     "<item>{:.4f}</item> "
     "<item>{:.4f}</item> "
     "<item>{:.4f}</item> "
     "<item>{:.4f}</item> "
     "<item>{:.4f}</item> "
     "<item>{:.4f}</item> "
     "<item>{:.4f}</item> "
     "</m-channelValues-0to1> "
     "</pControlInputs> "
     "</ExchangeData> "
     "</soap:Body> "
     "</soap:Envelope>"),
}


def is_number(s: str) -> bool:
    return s.replace('.', '').replace('-', '').isdigit()


def parse_tail(s: str):
    if is_number(s):
        return float(s)
    elif s == 'true':
        return True
    elif s == 'false':
        return False
    else:
        return s


def utf8len(s: str) -> int:
    """Return byte length of string.
    """

    return len(s.encode('utf-8'))


class FlightAxisConnector(object):
    """Simulator connector for FlightAxis Link

    """

    _URL = "192.168.0.5"
    _PORT = 18083
    _msg = ("POST / HTTP/1.1\n"
            "soapaction: '{}'\n"
            "content-length: {}\n"
            "content-type: text/xml;charset='UTF-8'\n"
            "Connection: Keep-Alive\n"
            "\n"
            "{}")

    def __init__(self):
        # self._host_ip = socket.gethostbyname(
        #     FlightAxisConnector._URL)  # type: str
        # self._socket = socket.socket(socket.AF_INET,
        #                              socket.SOCK_STREAM)  # type: socket.socket
        # # self._socket.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
        # # self._host_ip = "127.0.0.1"
        # self._socket.connect((self._host_ip, FlightAxisConnector._PORT))
        # self._socket.setblocking(0)

        self.servos = np.zeros(12, dtype=float)
        self.controller_started = False
        self.frame_counter = 0
        self.activation_frame_counter = 0
        self.average_frame_time_s = 0
        self.socket_frame_counter = 0
        self.state = {
            "rcin0": 0.5,
            "rcin1": 0.5,
            "rcin2": 0.0,
            "rcin3": 0.5,
            "rcin4": 0.0,
            "rcin5": 0.0,
            "rcin6": 0.0,
            "rcin7": 0.0,
            "rcin8": 0.0,
            "rcin9": 0.0,
            "rcin10": 0.0,
            "rcin11": 0.0,
            "m-currentPhysicsTime-SEC": 0,
            "m-currentPhysicsSpeedMultiplier": 1,
            "m-airspeed-MPS": 0,
            "m-altitudeASL-MTR": 0,
            "m-altitudeAGL-MTR": 0,
            "m-groundspeed-MPS": 0,
            "m-pitchRate-DEGpSEC": 0,
            "m-rollRate-DEGpSEC": 0,
            "m-yawRate-DEGpSEC": 0,
            "m-azimuth-DEG": 0,
            "m-inclination-DEG": 0,
            "m-roll-DEG": 0,
            "m-orientationQuaternion-X": 0,
            "m-orientationQuaternion-Y": 0,
            "m-orientationQuaternion-Z": 0,
            "m-orientationQuaternion-W": 0,
            "m-aircraftPositionX-MTR": 0,
            "m-aircraftPositionY-MTR": 0,
            "m-velocityWorldU-MPS": 0,
            "m-velocityWorldV-MPS": 0,
            "m-velocityWorldW-MPS": 0,
            "m-velocityBodyU-MPS": 0,
            "m-velocityBodyV-MPS": 0,
            "m-velocityBodyW-MPS": 0,
            "m-accelerationWorldAX-MPS2": 0,
            "m-accelerationWorldAY-MPS2": 0,
            "m-accelerationWorldAZ-MPS2": 0,
            "m-accelerationBodyAX-MPS2": 0,
            "m-accelerationBodyAY-MPS2": 0,
            "m-accelerationBodyAZ-MPS2": 0,
            "m-windX-MPS": 0,
            "m-windY-MPS": 0,
            "m-windZ-MPS": 0,
            "m-propRPM": 0,
            "m-heliMainRotorRPM": -1,
            "m-batteryVoltage-VOLTS": -1,
            "m-batteryCurrentDraw-AMPS": -1,
            "m-batteryRemainingCapacity-MAH": -1,
            "m-fuelRemaining-OZ": 0,
            "m-isLocked": False,
            "m-hasLostComponents": False,
            "m-anEngineIsRunning": True,
            "m-isTouchingGround": True,
            "m-flightAxisControllerIsActive": False,
            "m-currentAircraftStatus": "CAS-FLYING",
            "m-resetButtonHasBeenPressed": False,
        }

    def exchange_data(self, control_input) -> None:
        # if (not self.controller_started
        #         or self.state["m-flightAxisControllerIsActive"] == False
        #         or self.state["m-resetButtonHasBeenPressed"]):
        #     self.soap_request("RestoreOriginalControllerDevice")
        #     self.soap_request("InjectUAVControllerInterface")
        #     self.activation_frame_counter = self.frame_counter
        #     self.controller_started = True

        action = "ExchangeData"
        servo = control_input.tolist()
        ex_data_msg = ACTION_FMT[action].format(*servo)

        reply = self.soap_request(action, ex_data_msg)

        if reply:
            lastt_s = self.state["m-currentPhysicsTime-SEC"]
            self.parse_reply(reply)
            dt = self.state["m-currentPhysicsTime-SEC"] - lastt_s
            if 0 < dt < 0.1:
                if self.average_frame_time_s < 1e-6:
                    self.average_frame_time_s = dt
                self.average_frame_time_s = (self.average_frame_time_s * 0.98 +
                                             dt * 0.02)
            self.socket_frame_counter += 1

    def parse_reply(self, reply: str) -> None:
        split = reply.split(b'\n')[-2:]
        xml_txt = b"".join(split)
        # parser = ET.XMLParser(encoding="utf-8")
        root = ET.fromstring(xml_txt)
        some_data = root[0][0][0]
        for stuff in some_data:
            if(stuff.tag == "m-channelValues-0to1"):
                counter = 0;
                for item in stuff:
                    if item.tag == "item":
                        state_tag = "rcin"+str(counter)
                        counter += 1
                        self.state[state_tag] = parse_tail(item.text)

        aircraft_state = root[0][0][1]
        for item in aircraft_state:
            if item.tag in self.state.keys():
                self.state[item.tag] = parse_tail(item.text)

        notification = root[0][0][2]
        for item in notification:
            if item.tag in self.state.keys():
                self.state[item.tag] = parse_tail(item.text)

    def soap_request(self, action: str, body: str = "") -> str:
        url = "http://192.168.0.5:18083"
        _msg = ("POST / HTTP/1.1\n"
                "soapaction: '{}'\n"
                "content-length: {}\n"
                "content-type: text/xml;charset='UTF-8'\n"
                "Connection: Keep-Alive\n"
                "\n"
                "{}")
        body_encoded = body.encode("utf_8", errors="strict")
        headers = {"soapaction": action,
                "content-type": "text/xml;charset='UTF-8'",
                "Connection": "Keep-Alive",
                }

        response = requests.post(url, data=body, headers=headers)
        return response.content

    def soap_request2(self, action: str, fmt: str = "") -> str:
        """

        Args:
            action:
                "RestoreOriginalControllerDevice"
                "InjectUAVControllerInterface"
                "ExchangeData"
        Returns:
            Return string from RealFlight.
        """
        # _socket = socket.socket(socket.AF_INET,
        #                              socket.SOCK_STREAM)  # type: socket.socket
        # _socket.connect((self._host_ip, FlightAxisConnector._PORT))
        # _socket.setblocking(0)
        # _socket.settimeout(1)

        if fmt == "":
            fmt = ACTION_FMT[action]
        msg = FlightAxisConnector._msg.format(action, len(fmt), fmt)
        # ret = self._socket.send(bytes(msg, encoding="utf_8"))
        ret = self._socket.send(msg.encode("utf_8", errors="strict"))
        #print(bytes(msg, encoding="utf8"))

        self._socket.settimeout(1)
        chunk = self._socket.recv(4096)
        byte_received = len(chunk)
        data = chunk.decode()
        if data == 0:
            print("No data")
            return None

        # get content length
        m = re.search(r"(?<=Content-Length:).+", data)
        if m is None:
            print("No Content-Length")
            return None

        content_length = int(m.group(0))

        q = data.find("<?xml")
        print("DEBUG DEBUG DEBUG")
        print(data[q:])
        # _socket.close()
        return data

def mode_manual(_fac):
    control = np.zeros(12)
    for i in range(12):
        tag = "rcin" + str(i)
        control[i] = _fac.state[tag]
    return control

if __name__ == "__main__":
    fac = FlightAxisConnector()
    action = "RestoreOriginalControllerDevice"
    fac.soap_request(action, ACTION_FMT[action])
    action = "InjectUAVControllerInterface"
    fac.soap_request(action, ACTION_FMT[action])

    control_input = np.zeros(12)
    while 1:
        try:
            control_input = mode_manual(fac)

            fac.exchange_data(control_input)
        except KeyboardInterrupt:
            control_input[2] = 0
            control_input[0] = 0.5
            control_input[1] = 0.5
            control_input[3] = 0.5
            face.exchange_data(control_input)
        except:
            pass

