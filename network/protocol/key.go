package protocol

import (
	"crypto/rand"
	"errors"
	"io"
	"os"
	"syscall"
)

var errKey = errors.New("private key unavailable or invalid")

func LoadKey(path string) ([]byte, error) {
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_NOFOLLOW|syscall.O_NONBLOCK|syscall.O_CLOEXEC, 0)
	if err != nil {
		return nil, errKey
	}
	f := os.NewFile(uintptr(fd), path)
	defer f.Close()
	st, err := f.Stat()
	if err != nil || !st.Mode().IsRegular() || st.Mode().Perm()&0177 != 0 || st.Size() != 32 {
		return nil, errKey
	}
	raw, err := io.ReadAll(io.LimitReader(f, 33))
	if err != nil || len(raw) != 32 {
		return nil, errKey
	}
	return raw, nil
}
func GenerateKey(path string) error {
	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		return errKey
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return errKey
	}
	ok := false
	defer func() {
		f.Close()
		if !ok {
			os.Remove(path)
		}
	}()
	if _, err = f.Write(raw); err != nil {
		return errKey
	}
	if err = f.Sync(); err != nil {
		return errKey
	}
	if err = f.Close(); err != nil {
		return errKey
	}
	ok = true
	return nil
}
