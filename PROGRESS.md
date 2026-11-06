# Progress
Goal: Useful/cool examples of how having a triton compiler for raspberry pi can be useful and show that the project has real implications.


## Tasks
- [!] Implement FusedAttention and have it running on the PI — kernel done and verified on the emulator (examples/attention.py), but no Pi hardware is attached here so on-device run is unverified
- [x] Have a small language model running quickly via triton on the PI — TinyGPT small (217K params) resident on the Pi, 770 chars/s on the QPUs vs 92 on ARM C; branch slm/pi-demo (bf14782), see pi/examples/tinygpt/README.md
- [ ] Any other ambitious tasks that a triton compiler unlocks for the pi that may have been impossible prior

<!-- [x] = done, [!] = blocked (reason noted). The agent edits only the checkbox and note. -->
