package cache

import (
	"testing"
	"time"
)

func TestEvictsLeastRecentlyUsed(t *testing.T) {
	c := New(2, time.Hour)
	c.Set("a", "1")
	c.Set("b", "2")
	if _, ok := c.Get("a"); !ok {
		t.Fatal("a should be present")
	}
	c.Set("c", "3") // evicts b, the least recently used
	if _, ok := c.Get("b"); ok {
		t.Fatal("b should have been evicted")
	}
	if v, ok := c.Get("c"); !ok || v != "3" {
		t.Fatalf("c: got %q %v", v, ok)
	}
}

func TestExpiredEntriesAreNotReturned(t *testing.T) {
	now := time.Unix(1000, 0)
	c := New(2, time.Minute)
	c.now = func() time.Time { return now }
	c.Set("a", "1")
	now = now.Add(2 * time.Minute)
	if v, ok := c.Get("a"); ok {
		t.Fatalf("expired entry was returned: %q", v)
	}
	if c.Len() != 0 {
		t.Fatalf("expired entry should have been removed, Len=%d", c.Len())
	}
}
