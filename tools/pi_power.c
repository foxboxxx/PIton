// Switch the Pi's power through the HUGE UART board (CS340LX lab 3): the
// CP2102N's GPIO0 drives the MOSFET on the Pi's 5 V rail.  Same vendor request
// as the course's cp210x.c (bRequest 0xFF, wValue 0x37E1 = write latch,
// wIndex = mask | value << 8), sent through IOKit so libusb is not needed.
//
//   pi_power status          print the GPIO latch
//   pi_power on | off        set GPIO0 high / low
//   pi_power cycle [ms]      off, wait (default 500 ms), on
//
// Build: clang -O2 -o tools/pi_power tools/pi_power.c -framework IOKit -framework CoreFoundation
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOCFPlugIn.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/usb/IOUSBLib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CP210X_VID 0x10c4
#define CP210X_PID 0xea60
#define CP210X_VENDOR_SPECIFIC 0xff
#define CP210X_WRITE_LATCH 0x37e1
#define CP210X_READ_LATCH 0x00c2
#define GPIO0_MASK 0x0001

static IOUSBDeviceInterface **open_device(void) {
    static const char *classes[] = {"IOUSBHostDevice", kIOUSBDeviceClassName};
    for (int c = 0; c < 2; c++) {
        CFMutableDictionaryRef match = IOServiceMatching(classes[c]);
        int vid = CP210X_VID, pid = CP210X_PID;
        CFNumberRef v = CFNumberCreate(NULL, kCFNumberIntType, &vid);
        CFNumberRef p = CFNumberCreate(NULL, kCFNumberIntType, &pid);
        CFDictionarySetValue(match, CFSTR(kUSBVendorID), v);
        CFDictionarySetValue(match, CFSTR(kUSBProductID), p);
        CFRelease(v);
        CFRelease(p);
        io_service_t service = IOServiceGetMatchingService(kIOMainPortDefault, match);
        if (!service)
            continue;
        IOCFPlugInInterface **plugin = NULL;
        SInt32 score;
        kern_return_t kr = IOCreatePlugInInterfaceForService(service, kIOUSBDeviceUserClientTypeID,
                                                             kIOCFPlugInInterfaceID, &plugin, &score);
        IOObjectRelease(service);
        if (kr != kIOReturnSuccess || !plugin)
            continue;
        IOUSBDeviceInterface **dev = NULL;
        (*plugin)->QueryInterface(plugin, CFUUIDGetUUIDBytes(kIOUSBDeviceInterfaceID), (LPVOID *)&dev);
        (*plugin)->Release(plugin);
        if (dev)
            return dev;     // control requests on the default pipe need no USBDeviceOpen
    }
    fprintf(stderr, "pi_power: no CP2102N (%04x:%04x) found\n", CP210X_VID, CP210X_PID);
    return NULL;
}

static int request(IOUSBDeviceInterface **dev, int in, UInt16 value, UInt16 index, void *data, UInt16 len) {
    IOUSBDevRequest r = {
        .bmRequestType = USBmakebmRequestType(in ? kUSBIn : kUSBOut, kUSBVendor, kUSBDevice),
        .bRequest = CP210X_VENDOR_SPECIFIC,
        .wValue = value,
        .wIndex = index,
        .wLength = len,
        .pData = data,
    };
    IOReturn rc = (*dev)->DeviceRequest(dev, &r);
    if (rc != kIOReturnSuccess) {
        fprintf(stderr, "pi_power: control request failed: 0x%08x\n", rc);
        return -1;
    }
    return 0;
}

static int gpio0_set(IOUSBDeviceInterface **dev, int high) {
    return request(dev, 0, CP210X_WRITE_LATCH, GPIO0_MASK | (high ? GPIO0_MASK << 8 : 0), NULL, 0);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: pi_power status | on | off | cycle [ms]\n");
        return 2;
    }
    IOUSBDeviceInterface **dev = open_device();
    if (!dev)
        return 1;
    int rc = 0;
    if (!strcmp(argv[1], "status")) {
        unsigned char latch = 0;
        rc = request(dev, 1, CP210X_READ_LATCH, 0, &latch, 1);
        if (!rc)
            printf("latch 0x%02x (GPIO0 %s)\n", latch, latch & GPIO0_MASK ? "high" : "low");
    } else if (!strcmp(argv[1], "on") || !strcmp(argv[1], "off")) {
        rc = gpio0_set(dev, !strcmp(argv[1], "on"));
    } else if (!strcmp(argv[1], "cycle")) {
        int ms = argc > 2 ? atoi(argv[2]) : 500;
        rc = gpio0_set(dev, 0);
        usleep((useconds_t)ms * 1000);
        rc |= gpio0_set(dev, 1);
    } else {
        fprintf(stderr, "pi_power: unknown command %s\n", argv[1]);
        rc = 2;
    }
    (*dev)->Release(dev);
    return rc ? 1 : 0;
}
