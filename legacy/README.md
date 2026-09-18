# Legacy C++ CLI workflows

These file-based workflows use the native `Analysis` and `Synthesis`
executables. For the recommended in-memory Python API, see the main
[README](../README.md).

The examples assume 16 kHz audio. Other sampling rates are supported, but the
configuration must be changed accordingly.

## Example data

Download a wave file from the Arctic database:

```bash
URL='http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_slt_arctic/wav/arctic_a0001.wav'
DATADIR='./data/tmp'
BASENAME='slt_arctic_a0001'
mkdir -p "$DATADIR"
curl -L -o "$DATADIR/$BASENAME.wav" "$URL"
```

## Acoustic feature analysis

Run the legacy analysis program with the default configuration:

```bash
Analysis "$DATADIR/$BASENAME.wav" ./config/config_default_16k.cfg
```

The analysis produces files such as:

```text
./data/tmp/slt_arctic_a0001.gain
./data/tmp/slt_arctic_a0001.lsf
./data/tmp/slt_arctic_a0001.slsf
./data/tmp/slt_arctic_a0001.hnr
./data/tmp/slt_arctic_a0001.pls
./data/tmp/slt_arctic_a0001.f0
./data/tmp/slt_arctic_a0001.src.wav
```

## Synthesis with single-pulse excitation

Run copy synthesis with the default single-pulse excitation:

```bash
Synthesis "$DATADIR/$BASENAME" ./config/config_default_16k.cfg
mv "$DATADIR/$BASENAME.syn.wav" "$DATADIR/$BASENAME.syn.sp.wav"
```

The output is `./data/tmp/slt_arctic_a0001.syn.sp.wav`.

## Synthesis with original pulses

The extracted pulses can be reassembled with pitch-synchronous overlap-add.
Create a user configuration that selects pulse-as-feature excitation:

```bash
CONF_USR="$DATADIR/config_usr.cfg"
echo '# Comment: User config for GlottDNN' > "$CONF_USR"
echo 'EXCITATION_METHOD = "PULSES_AS_FEATURES";' >> "$CONF_USR"
echo 'USE_WSOLA = true;' >> "$CONF_USR"
echo 'USE_SPECTRAL_MATCHING = false;' >> "$CONF_USR"
echo 'NOISE_GAIN_VOICED = 0.0;' >> "$CONF_USR"
```

Run synthesis with both configuration files:

```bash
Synthesis "$DATADIR/$BASENAME" ./config/config_default_16k.cfg "$CONF_USR"
mv "$DATADIR/$BASENAME.syn.wav" "$DATADIR/$BASENAME.syn.paf.wav"
```

The original pulses are not available in many applications, such as text to
speech; a trainable excitation model can be used in those cases.
