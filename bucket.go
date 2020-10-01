package cuckoo

import (
	"bytes"
	"fmt"
)

type fingerprint uint16

type bucket [bucketSize]fingerprint

const (
	nullFp              = 0
	bucketSize          = 4
	fingerprintSizeBits = 16
)

func (b *bucket) insert(fp fingerprint) bool {
	for i, tfp := range b {
		if tfp == nullFp {
			b[i] = fp
			return true
		}
	}
	return false
}

func (b *bucket) delete(fp fingerprint) bool {
	for i, tfp := range b {
		if tfp == fp {
			b[i] = nullFp
			return true
		}
	}
	return false
}

func (b *bucket) getFingerprintIndex(fp fingerprint) int {
	for i, tfp := range b {
		if tfp == fp {
			return i
		}
	}
	return -1
}

func (b *bucket) reset() {
	for i := range b {
		b[i] = nullFp
	}
}

func (b *bucket) String() string {
	var buf bytes.Buffer
	buf.WriteString("[")
	for _, by := range b {
		buf.WriteString(fmt.Sprintf("%5d ", by))
	}
	buf.WriteString("]")
	return buf.String()
}