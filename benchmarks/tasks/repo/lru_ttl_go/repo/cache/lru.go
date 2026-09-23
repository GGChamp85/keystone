// Package cache is a small LRU cache whose entries expire after a TTL.
package cache

import (
	"container/list"
	"time"
)

type entry struct {
	key     string
	value   string
	expires time.Time
}

// Cache holds at most `capacity` entries; each entry expires `ttl` after it was set.
type Cache struct {
	capacity int
	ttl      time.Duration
	now      func() time.Time
	order    *list.List
	items    map[string]*list.Element
}

func New(capacity int, ttl time.Duration) *Cache {
	return &Cache{capacity: capacity, ttl: ttl, now: time.Now, order: list.New(), items: map[string]*list.Element{}}
}

// Set stores value under key, evicting the least recently used entry when full.
func (c *Cache) Set(key, value string) {
	if el, ok := c.items[key]; ok {
		el.Value.(*entry).value = value
		el.Value.(*entry).expires = c.now().Add(c.ttl)
		c.order.MoveToFront(el)
		return
	}
	if c.order.Len() >= c.capacity {
		oldest := c.order.Back()
		if oldest != nil {
			c.order.Remove(oldest)
			delete(c.items, oldest.Value.(*entry).key)
		}
	}
	el := c.order.PushFront(&entry{key: key, value: value, expires: c.now().Add(c.ttl)})
	c.items[key] = el
}

// Get returns the value and true when key is present and not expired.
func (c *Cache) Get(key string) (string, bool) {
	el, ok := c.items[key]
	if !ok {
		return "", false
	}
	// BUG: the expiry is never checked, so a stale entry is returned (and refreshed as most recently used).
	c.order.MoveToFront(el)
	return el.Value.(*entry).value, true
}

// Len is the number of entries currently held, expired or not.
func (c *Cache) Len() int { return c.order.Len() }
